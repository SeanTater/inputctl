from reflex_train.data.intent import infer_intent_from_keys


class KeyWindowIntentLabeler:
    def __init__(self, intent_horizon: int = 0):
        self.intent_horizon = intent_horizon

    def label_intents(self, video_path, frame_indices, key_set_by_frame):
        intents = []
        for i in range(len(frame_indices)):
            end = min(i + self.intent_horizon, len(frame_indices) - 1)
            window = key_set_by_frame[i:end + 1]
            union_keys = set().union(*window) if window else set()
            if not union_keys:
                intents.append("WAIT")
            else:
                intents.append(infer_intent_from_keys(union_keys))
        return intents
