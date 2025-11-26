class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        # nn.Sequential을 사용하여 층을 쌓습니다.
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),  # 입력층 -> 은닉층
            nn.ReLU(),                           # 활성화 함수
            nn.Linear(hidden_size, hidden_size), # 은닉층 -> 은닉층 (추가 가능)
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)  # 은닉층 -> 출력층
        )

    def forward(self, x):
        # MLP는 시계열 차원(Sequence Length)이 보통 없으므로 
        # (Batch, Seq, Feature) 형태라면 (Batch, Feature)로 평탄화(flatten)할 수도 있습니다.
        # 여기서는 단순하게 (Batch, Input_Size)가 들어온다고 가정합니다.
        return self.network(x)
