class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # batch_first=True: 입력 데이터 형태를 (Batch, Seq, Feature)로 설정
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        
        # 마지막 시점의 출력을 원하는 output_size로 변환하기 위한 선형 층
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        
        # 초기 은닉 상태 (0으로 초기화)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # RNN 순전파
        # out shape: (batch, seq, hidden_size)
        # hn shape: (num_layers, batch, hidden_size)
        out, hn = self.rnn(x, h0)
        
        # 다대일(Many-to-One) 구조: 마지막 시점(Step)의 결과만 사용하여 예측
        last_time_step_out = out[:, -1, :] 
        return self.fc(last_time_step_out)
