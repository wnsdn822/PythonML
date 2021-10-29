from sklearn.feature_extraction.text import CountVectorizer
from konlpy.tag import Okt
import re
okt=Okt()

# .제거
token=re.sub("(\.)","","한국폴리텍대학 서울강섴맴퍼스 데이터분석과 이협건 교수는 "
                       "한국폴리텍대학에서 데이터분석 과목과 인공지능 과목을 교육하는 교수이다.")

# 형태소 단위로 나누기
token=okt.morphs(token)

# 형태소 단위로 분리된 벡터 데이터 구조를 문장 형태로 변환
# 문장 형태는 벡터마다 띄어씌기 추가함
corpus = [" ".join(token)]

# CountVectorizer는 Bow 알고리즘의 단어의 단위를 띄어쓰기 한칸으로 구분함
vector = CountVectorizer()

# 코퍼스로부터 각 단어의 빈도 수를 기록한다
print(vector.fit_transform(corpus).toarray())

# 각 단어의 인덱스가 어떻게 부여되었는지를 보여준다.
print(vector.vocabulary_)