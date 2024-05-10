import torch
import time
import matplotlib.pyplot as plt

# 입력값 설정
N, d = 10, 4  # 토큰의 수 N, 채널 차원 d
d_range = 2  # 주변 토큰 범위

# Query와 Key 텐서 초기화
Query = torch.randn(N, d)
Key = torch.randn(N, d)

# 어텐션 스코어를 저장할 텐서 초기화
attention_scores = torch.zeros(N, N)

# 마스킹 행렬 생성
mask = torch.zeros(N, N)
for i in range(N):
    for j in range(max(0, i-d_range), min(N, i+d_range+1)):
        mask[i, j] = 1

# 희소 행렬 곱셈 수행
masked_key = mask @ Key
# start_time = time.time()
dilated_scores = torch.matmul(Query, masked_key.T)
# dilate_delay1 = time.time() - start_time
# print("Dilated attention: --- %s seconds ---" % (dilate_delay1))

# start_time = time.time()
normal_scores = Query @ Key.transpose(0, 1)
# dilate_delay2 = time.time() - start_time
# print("Normal attention: --- %s seconds ---" % (dilate_delay2))

# # 결과 텐서 시각화
# plt.imshow(attention_scores, cmap='viridis')
# plt.colorbar(label='Attention Score')
# plt.xlabel('Key Tokens')
# plt.ylabel('Query Tokens')
# plt.title('Attention Scores for Adjacent Key Tokens')
# plt.show()





# def dilated_weighted_dot_product(Q, K, dilation_rate=3, weights=None):
#     N, d = Q.size()
#     scores = torch.zeros(N, N)
    
#     for i in range(N):
#         for j in range(0, N, dilation_rate):
#             if i == j:  # 가까운 토큰에 대한 처리
#                 weight = weights[0] if weights else 1.0
#             else:  # 멀리 떨어진 토큰에 대한 처리
#                 weight = weights[1] if weights else 0.5
#             scores[i, j] = (Q[i] * K[j]).sum() * weight
    
#     return scores

# # 예제 사용
# N, d = 10, 64  # 10개의 토큰과 64의 채널 차원
# Q = torch.rand(N, d)
# K = torch.rand(N, d)

# # 딜레이션 비율과 가중치 설정
# dilation_rate = 3
# weights = [1.0, 0.5]  # 가까운 토큰에는 1.0, 멀리 떨어진 토큰에는 0.5의 가중치

# start_time = time.time()
# scores = dilated_weighted_dot_product(Q, K, dilation_rate, weights)
# dilate_delay1 = time.time() - start_time
# print("Dilated attention: --- %s seconds ---" % (dilate_delay1))

# start_time = time.time()
# normal_scores = Q @ K.transpose(0, 1)
# dilate_delay2 = time.time() - start_time
# print("Normal attention: --- %s seconds ---" % (dilate_delay2))

# ratio = dilate_delay1/dilate_delay2
