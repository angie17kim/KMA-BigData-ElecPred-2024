### 코드 설명

1. configs/base.yaml에서 경로 정보를 설정합니다.
2. preprocessing 디렉토리 내부의 노트북을 순서대로 지시에 따라 실행합니다.
3. `python3 run.py --config units.yaml` (혹은 `no_units`) 로 실행합니다. (최종 제출버전에 해당하는 config 입니다)
4. 지정한 output_path에서 결과를 확인합니다.

### 기타
- 전처리 노트북 1에서 한국의 공휴일을 불러오기 위하여 API키가 필요한데 개인적으로 사용한 키는 제거하였습니다. 참고 웹사이트를 참고하여 재설정해주세요.
- 0_check_submission.ipynb와 1_ensemble.ipynb는 제출을 위하여 사용한 코드로 경로가 하드코딩되어있습니다. 대응하는 경로로 변경해주세요.