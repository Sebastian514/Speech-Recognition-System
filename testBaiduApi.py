
def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()
from aip import AipSpeech
APP_ID ='24917387'
API_KEY = '4DzTBjliiLURreWKIbfCkdRf'
SECRET_KEY = 'uv0XPf1iUYks6DT0yBw1lwuYhsVTmNRG'
client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)
#print(get_file_content('./16k.wav'))
print(client.asr(get_file_content('./firstspeech.wav'), 'wav', 8000, {
    'dev_pid': 1537,
}))
