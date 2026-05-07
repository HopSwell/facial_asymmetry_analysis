import pyttsx3

engine = pyttsx3.init()
voices = engine.getProperty('voices')

print("사용 가능한 목소리 목록:")
found_korean = False
for voice in voices:
    print(f"- {voice.name}")
    if "Korea" in voice.name or "KR" in voice.id:
        engine.setProperty('voice', voice.id)
        found_korean = True

if found_korean:
    print("\n한국어 목소리를 찾았습니다! 테스트합니다.")
    engine.say("안녕하세요 제 이름은 전승혁입니다")
else:
    print("\n한국어 목소리가 없습니다. 영어로 테스트합니다.")
    engine.say("Hello my name is Jeon")

engine.runAndWait()