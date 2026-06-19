import os

# 함수정의
# 디렉토리(폴더)명과 파일 바이트, 원본 파일명을 알려주면,
# 해당 디렉토리에 파일을 저장해 주는 함수. (FastAPI UploadFile 기준)
def save_uploaded_file(directory, filename, content: bytes):
    # 1. 디렉토리가 있는지 확인하여, 없으면 먼저, 디렉토리부터 만든다.
    if not os.path.exists(directory):
        os.makedirs(directory)

    # 2. 디렉토리가 있으니, 파일을 저장한다.
    file_path = os.path.join(directory, filename)
    with open(file_path, 'wb') as f:
        f.write(content)

    # 3. 파일 저장이 성공했으니, 경로를 리턴
    return file_path
