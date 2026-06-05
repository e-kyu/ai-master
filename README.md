## 실행환경
- python3.12 버전
- 의존성 패키지는 각 하위 프로젝트의 "requirements.txt"파일 참조

## 실행방법
```cmmand
python ./llm-server/main.py
```

## docekrfile (docker image 생성)
```cmmand
docker build -t kimbab-ax/server .
```

## docekr-compose (docker 실행)
```cmmand
docker-compose up -d
```
