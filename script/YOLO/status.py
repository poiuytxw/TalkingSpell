import json
tts_clone=["0_old_woman","1_young_woman","2_kid_woman","3_old_man","4_young_man","5_kid_man","6_no_gender"]
class InfoStorage:
    def __init__(self, filename='../../m_settings/m_status.json'):
        self.filename = filename
        self.load()

    def load(self):
        try:
            with open(self.filename, 'r') as f:
                data = json.load(f)
                self.detection = data.get('detection', None)
                self.char_path = data.get('char_path', None)
                self.tts_path = data.get('tts_path', None)
                self.chat_history=data.get('chat_history', None)
                self.tts_selection=data.get('tts_selection', None)
        except FileNotFoundError:
            self.detection = None
            self.char_path = None
            self.tts_path = None
            self.chat_history = None
            self.tts_selection = None

    def update_detection(self, new_detect,new_tts):
        self.detection = new_detect

        if new_detect=="nothing":
            self.char_path=self.tts_path=self.tts_selection=self.chat_history="unknown"
        else:
            self.char_path=f'../../m_settings/char_{new_detect}.json'
            self.chat_history=f'../../m_settings/messages_{new_detect}.json'
            self.tts_selection=new_tts
            self.tts_path="../../m_voice/cloning_audio/"+tts_clone[new_tts]+".mp3"
        self.save()

    def update_info2(self, new_info):
        self.info2 = new_info
        self.save()

    def save(self):
        with open(self.filename, 'w') as f:
            json.dump({"detection": self.detection, 
                        "char_path":self.char_path,
                        "chat_history":self.chat_history,
                        "tts_selection":self.tts_selection,
                        "tts_path": self.tts_path}, f)

    def get_detection(self):
        self.load()
        return self.detection
    
    def get_history(self):
        self.load()
        return self.chat_history

    def get_tts(self):
        self.load()
        return self.tts_path
    
    def get_char(self):
        self.load()
        return self.char_path