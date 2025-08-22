
def init(context):
    print("Init okay")

def serve(context, event):
    return {"status": "Okay", "received_headers": event.headers}
