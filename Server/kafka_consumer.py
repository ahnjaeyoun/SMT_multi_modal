import time
import json
from kafka import KafkaProducer
from kafka.errors import KafkaError

# Tailscale VPN을 통해 접속할 외부망 Kafka 브로커 IP 및 포트
# 예: Tailscale IP가 100.101.102.103 이라면 '100.101.102.103:9092'
KAFKA_BROKER = '100.70.106.105:9092'  
KAFKA_TOPIC = 'test_topic'

def test_kafka_connection():
    """Tailscale VPN을 통한 Kafka 브로커 통신 상태를 테스트합니다."""
    print(f"Tailscale VPN 망을 통해 Kafka 브로커({KAFKA_BROKER}) 접속 시도 중...")
    
    try:
        # Producer 생성 시도
        # 연결이 되지 않으면 타임아웃이 발생합니다.
        producer = KafkaProducer(
            bootstrap_servers=[KAFKA_BROKER],
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            # 빠른 연결 실패 확인을 위해 타임아웃 시간을 10초로 설정
            request_timeout_ms=10000,
            max_block_ms=10000
        )
        print("Kafka 브로커 연결 성공!")

        # 테스트 메시지 구성
        test_message = {
            "type": "ping",
            "message": "Tailscale VPN 연결 테스트 메시지입니다.",
            "timestamp": time.time()
        }

        print(f"토픽 '{KAFKA_TOPIC}'으로 테스트 메시지 전송 시도...")
        # 메시지 전송
        future = producer.send(KAFKA_TOPIC, value=test_message)
        
        # 메시지 전송 결과를 동기식으로 기다려서 확인 (최대 10초)
        record_metadata = future.get(timeout=10)
        
        print("메시지 전송 성공!")
        print(f"   - 전송된 토픽: {record_metadata.topic}")
        print(f"   - 파티션 번호: {record_metadata.partition}")
        print(f"   - 오프셋: {record_metadata.offset}")

        producer.flush()
        producer.close()
        print("테스트 완료 및 연결 정상 종료")

    except KafkaError as e:
        print(f"Kafka 통신 오류 (방화벽, IP, 포트, 혹은 Kafka 서버 상태 확인 필요): {e}")
    except Exception as e:
        print(f"예기치 않은 오류 발생: {e}")

if __name__ == "__main__":
    print("Kafka 연결 테스트 스크립트 시작...")
    test_kafka_connection()