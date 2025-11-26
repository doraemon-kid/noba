if __name__ == "__main__":
    # 하이퍼파라미터 설정
    INPUT_SIZE = 10    # 입력 특성(Feature)의 개수
    HIDDEN_SIZE = 32   # 은닉층 노드 개수
    OUTPUT_SIZE = 1    # 출력 개수 (예: 회귀 문제라면 1)
    SEQ_LENGTH = 5     # 시계열 데이터의 길이 (Time steps)
    BATCH_SIZE = 8     # 배치 크기

    print("=== 모델 테스트 시작 ===\n")

    # 1. MLP 테스트 (입력: [Batch, Input_Size])
    # 시계열 차원을 무시하고 하나의 벡터로 처리한다고 가정
    mlp_input = torch.randn(BATCH_SIZE, INPUT_SIZE)
    mlp_model = MLP(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    mlp_output = mlp_model(mlp_input)
    print(f"1. MLP (ANN) Output shape: {mlp_output.shape}")

    # 시계열 모델용 입력 데이터 (입력: [Batch, Seq_Length, Input_Size])
    seq_input = torch.randn(BATCH_SIZE, SEQ_LENGTH, INPUT_SIZE)

    # 2. RNN 테스트
    rnn_model = SimpleRNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    rnn_output = rnn_model(seq_input)
    print(f"2. RNN Output shape     : {rnn_output.shape}")

    # 3. LSTM 테스트
    lstm_model = LSTM(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    lstm_output = lstm_model(seq_input)
    print(f"3. LSTM Output shape    : {lstm_output.shape}")

    # 4. GRU 테스트
    gru_model = GRU(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    gru_output = gru_model(seq_input)
    print(f"4. GRU Output shape     : {gru_output.shape}")
    
    print("\n=== 모든 모델이 정상적으로 동작했습니다 ===")
