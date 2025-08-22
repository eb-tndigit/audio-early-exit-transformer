import os
import torch
from dataclasses import dataclass
from typing import List
import torch.nn.functional as F
from torchaudio.models.decoder import ctc_decoder, cuda_ctc_decoder


class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, blank=0):
        super().__init__()
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> List[str]:
        """Given a sequence emission over labels, get the best path
        Args:
        emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.
        Returns:
        List[str]: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        return indices


@dataclass
class Point:
    token_index: int
    time_index: int
    score: float


class BeamInference(object):

    def __init__(self, args):
        self.args = args

        # for bigger LM
        self.LM_WEIGHT = 1.0  # 3.23#1.0#3.23
        self.WORD_SCORE = -4  # -1.0#-0.26
        self.N_BEST = 1  # 500#300

        '''
        #for smaller LM
        self.LM_WEIGHT = 10.0
        self.WORD_SCORE = -0.26
        self.N_BEST = 1
        '''

        if args.bpe == True:
            self.decoder = []
            # for w_ins in [-1,-1,-1,-1,-1, -1]: #valori positivi aumentano le inserzioni
            for w_ins in [0, 0, 0, 0, 0, 0]:  # valori positivi aumentano le inserzioni
                # for w_ins in [-0.5,0.5,0.5,0.5,0.5, 0.5]:
                self.decoder += [ctc_decoder(lexicon=args.lexicon,
                                             tokens=args.tokens,
                                             nbest=self.N_BEST,
                                             log_add=False,
                                             beam_size=args.beam_size,  # 0, #500,
                                             word_score=w_ins,
                                             lm_weight=self.LM_WEIGHT,
                                             blank_token="@",
                                             unk_word="<unk>",
                                             sil_token="<pad>")]
        else:
            self.beam_search_decoder = ctc_decoder(
                lexicon=args.lexicon,
                tokens=args.tokens,
                nbest=1,
                log_add=True,
                beam_size=args.beam_size,
                lm_weight=self.LM_WEIGHT,
                word_score=self.WORD_SCORE
            )

        # Initialize CUDA decoder with H100 compatibility fixes
        self.cuda_decoder = None
        self.use_cuda_decoder = self._initialize_cuda_decoder(args)
        
        self.greedy_decoder = GreedyCTCDecoder()

    def _initialize_cuda_decoder(self, args):
        """Initialize CUDA decoder with H100 compatibility and fallback handling"""
        try:
            # Clear CUDA cache and set memory management for H100
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.set_per_process_memory_fraction(0.8)
                
            # Try with conservative parameters first
            self.cuda_decoder = cuda_ctc_decoder(
                args.tokens, 
                nbest=1, 
                beam_size=min(args.beam_size, 5),  # Limit beam size for H100
                blank_skip_threshold=0.95
            )
            print("CUDA CTC decoder initialized successfully with conservative settings")
            return True
            
        except Exception as e:
            print(f"Failed to initialize CUDA decoder: {e}")
            print("Will use CPU decoder as fallback")
            return False

    def beam_predict(self, model, input_sequence):
        emission = model.ctc_encoder(input_sequence)
        beam_search_result = self.beam_search_decoder(emission.cpu())
        beam_search_transcript = " ".join(
            beam_search_result[0][0].words).strip()
        return (beam_search_transcript)

    def ctc_predict_(self, emission, index=5):
        beam_search_result = self.decoder[index](emission.cpu())
        beam_search_transcript = []
        for s_ in beam_search_result:
            beam_search_transcript = beam_search_transcript + \
                [" ".join(s_[0].words).strip()]
        return (beam_search_transcript)

    def ctc_cuda_predict(self, emission, tokens=None):
        """CUDA CTC prediction with H100 compatibility and robust fallback"""
        if tokens is None:
            tokens = self.args.tokens
        
        # Ensure proper tensor properties
        emission = emission.contiguous()
        
        # Create length tensor
        enc_len = torch.full(
            size=(emission.size(0),), 
            fill_value=emission.size(1), 
            dtype=torch.int32
        ).to(self.args.device)
        
        if self.use_cuda_decoder and self.cuda_decoder is not None:
            try:
                # Method 1: Try CUDA decoder with synchronization
                torch.cuda.synchronize()
                
                # Ensure tensors are properly aligned
                emission_cuda = emission.contiguous()
                enc_len_cuda = enc_len.contiguous()
                
                results = self.cuda_decoder(emission_cuda, enc_len_cuda)
                print("CUDA decoder succeeded")
                return results
                
            except RuntimeError as cuda_error:
                if "cudaErrorIllegalAddress" in str(cuda_error):
                    print(f"CUDA decoder failed with memory error: {cuda_error}")
                    print("Falling back to CPU decoder...")
                    self.use_cuda_decoder = False  # Disable for future calls
                else:
                    raise cuda_error
                    
        # Fallback to CPU decoder
        try:
            print("Using CPU decoder fallback...")
            
            # Create CPU decoder on demand
            cpu_decoder = ctc_decoder(
                lexicon=None,
                tokens=tokens,
                nbest=1,
                log_add=False,
                beam_size=min(self.args.beam_size, 5),
                blank_token="@"
            )
            
            # Move tensors to CPU
            emission_cpu = emission.cpu()
            enc_len_cpu = enc_len.cpu()
            
            results = cpu_decoder(emission_cpu, enc_len_cpu)
            print("CPU decoder succeeded")
            return results
            
        except Exception as cpu_error:
            print(f"CPU decoder also failed: {cpu_error}")
            # Last resort: greedy decoding
            print("Using greedy decoder as last resort...")
            
            # Apply softmax and get greedy result
            probs = torch.nn.functional.softmax(emission, dim=-1)
            greedy_indices = self.greedy_decoder(probs.squeeze(0))
            
            # Create a mock result structure compatible with the expected format
            from types import SimpleNamespace
            mock_result = SimpleNamespace()
            mock_result.tokens = greedy_indices
            mock_result.score = 0.0
            
            return [[mock_result]]

    def ctc_predict(self, emission, index=5):
        beam_search_result = self.decoder[index](emission.cpu())
        beam_search_transcript = [
            " ".join(beam_search_result[0][0].words).strip()]
        nhyps = len(beam_search_result[0])
        hyp_score = torch.zeros(nhyps)

        for i in range(0, nhyps):
            hyp_score[i] = beam_search_result[0][i].score

        pprob = F.softmax(hyp_score, dim=0)
        return beam_search_transcript, pprob[0]

    def get_trellis(self, emission, tokens, blank_id=0):

        num_frame = emission.size(0)
        num_tokens = len(tokens)

        # Trellis has extra diemsions for both time axis and tokens.
        # The extra dim for tokens represents <SoS> (start-of-sentence)
        # The extra dim for time axis is for simplification of the code.
        trellis = torch.empty((num_frame + 1, num_tokens + 1)).to(self.args.device)
        trellis[0, 0] = 0
        trellis[1:, 0] = torch.cumsum(emission[:, 0], 0)
        trellis[0, -num_tokens:] = -float("inf")
        trellis[-num_tokens:, 0] = float("inf")

        for t in range(num_frame):
            trellis[t + 1, 1:] = torch.maximum(
                # Score for staying at the same token
                trellis[t, 1:] + emission[t, blank_id],
                # Score for changing to the next token
                trellis[t, :-1] + emission[t, tokens],
            )
        return trellis

    def backtrack(self, trellis, emission, tokens, blank_id=0):
        # Note:
        # j and t are indices for trellis, which has extra dimensions
        # for time and tokens at the beginning.
        # When referring to time frame index `T` in trellis,
        # the corresponding index in emission is `T-1`.
        # Similarly, when referring to token index `J` in trellis,
        # the corresponding index in transcript is `J-1`.
        j = trellis.size(1) - 1
        # t_start = torch.argmax(trellis[:, j]).item()
        t_start = trellis.size(0)-1
        path = []
        prob = 0
        for t in range(t_start, 0, -1):
            # 1. Figure out if the current position was stay or change
            # Note (again):
            # `emission[J-1]` is the emission at time frame `J` of trellis dimension.
            # Score for token staying the same from time frame J-1 to T.
            stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
            # Score for token changing from C-1 at T-1 to J at T.
            changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

            # 2. Store the path with frame-wise probability.
            # prob = emission[t - 1, tokens[j - 1] if changed > stayed else 0].exp().item()
            prob = prob + emission[t - 1, tokens[j - 1]
                                   if changed > stayed else 0].item()
            # Return token index and time index in non-trellis coordinate.
            path.append(Point(j - 1, t - 1, prob))

            # 3. Update the token

            if changed > stayed:
                j -= 1
                if j == 0:
                    break
        if j > 0:
            # raise ValueError("Failed to align")
            print(t, j, "Failed to align")
        return path[::-1]

    def sequence_length_penalty(self, length: int, alpha: float = 0.6) -> float:
        return ((5 + length) / (5 + 1)) ** alpha

    def beam_search(self, model, encoder_output, layer_n, 
                    vocab_size=None, max_length=500, min_length=300, 
                    SOS_token=None, EOS_token=None, PAD_token=None, 
                    beam_size=None, pen_alpha=None, return_best_beam=True):

        if vocab_size == None:
            vocab_size = self.args.dec_voc_size
        if SOS_token == None:
            SOS_token = self.args.trg_sos_idx
        if EOS_token == None:
            EOS_token = self.args.trg_eos_idx
        if PAD_token == None:
            PAD_token = self.args.trg_pad_idx
        if beam_size == None:
            beam_size = self.args.beam_size
        if pen_alpha == None:
            pen_alpha = self.args.pen_alpha
            
        beam_size_count = beam_size
        
        # decoder_input = input_decoder[:,0:input_decoder.size(1)-5]
        decoder_input = torch.tensor(
            [[SOS_token]], dtype=torch.long, device=self.args.device)

        scores = torch.Tensor([0.]).to(self.args.device)
        # print("DECODER_INPUT:",  text_transform.int_to_text(decoder_input.squeeze(0)))
        # input_sequence = input_sequence.to(device)

        # encoder_output = model._encoder_(input_sequence, valid_length, layer_n).to(device)

        # _,emission = model(input_sequence,decoder_input)
        final_scores = []
        final_tokens = []

        for i in range(max_length):
            # decoder_input = F.pad(decoder_input, (0,10), mode='constant',value=PAD_token)

            if i == 0:
                logits = model._decoder_(
                    decoder_input, encoder_output, layer_n).detach()
            else:
                logits = model._decoder_(decoder_input, encoder_output.expand(
                    beam_size_count, *encoder_output.shape[1:]), layer_n).detach()

            log_probs = logits[:, -1] / self.sequence_length_penalty(i+1, pen_alpha)
            scores = scores.unsqueeze(1) + log_probs
            scores, indices = torch.topk(scores.reshape(-1), beam_size_count)

            beam_indices = torch.divide(
                indices, vocab_size, rounding_mode='floor')
            token_indices = torch.remainder(indices, vocab_size)
            next_decoder_input = []
            EOS_beams_index = []

            for ind, (beam_index, token_index) in enumerate(zip(beam_indices, token_indices)):

                prev_decoder_input = decoder_input[beam_index]

                if token_index == EOS_token and i > min_length:

                    token_index = torch.LongTensor([token_index]).to(self.args.device)
                    final_tokens.append(
                        torch.cat([prev_decoder_input, token_index]))
                    final_scores.append(scores[ind])
                    beam_size_count -= 1

                    # scores_list = scores.tolist()
                    # del scores_list[ind]
                    # scores = torch.tensor(scores_list, device=device)
                    EOS_beams_index.append(ind)
                    # print(f"Beam #{ind} reached EOS!")

                else:
                    token_index = torch.LongTensor(
                        [token_index]).to(self.args.device)
                    next_decoder_input.append(
                        torch.cat([prev_decoder_input, token_index]))
            if len(EOS_beams_index) > 0:
                scores_list = scores.tolist()
                for tt in EOS_beams_index[::-1]:
                    del scores_list[tt]
                scores = torch.tensor(scores_list, device=self.args.device)

            if len(final_scores) == beam_size:
                break

            decoder_input = torch.vstack(next_decoder_input)

        # We have reached max # of allowed iterations.
        if i == (max_length - 1):

            for beam_unf, score_unf in zip(decoder_input, scores):
                final_tokens.append(beam_unf)
                final_scores.append(score_unf)
                del beam_unf
                del score_unf

            assert len(final_tokens) == beam_size and len(final_scores) == beam_size, (
                'Final_tokens and final_scores lists do not match beam_size size!')

        # If we want to return most probable predicted beam.
        # del encoder_output
        # del encoder_output_afterEOS
        del decoder_input
        del scores
        if return_best_beam:
            del encoder_output
            max_val = max(final_scores)

        return final_tokens, final_scores, final_tokens[final_scores.index(max_val)].tolist()