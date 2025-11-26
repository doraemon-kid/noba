class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 초기 은닉 상태(h0) 초기화 (GRU는 c0가 필요 없음)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, hn = self.gru(x, h0)
        
        # 마지막 시점의 출력 사용
        last_time_step_out = out[:, -1, :]
        return self.fc(last_time_step_out)
