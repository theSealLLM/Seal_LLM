import _common

from theseal.prompt import (
    build_label_specific_prompt,
    build_common_policy_prompt,
)


# 라벨별 질문과 프롬프트에 반드시 포함되어야 하는 정책 문구
TEST_CASES = [
    ("shipping", "배송이 언제 오나요?", "배송 정보는"),
    ("refund", "반품하고 싶어요.", "수령 후 7일"),
    ("account", "비밀번호를 잊었어요.", "/account/reset"),
]



def validate_prompt_builder(builder, expected_version):
    """프롬프트 생성 함수의 라벨, 질문, 정책 문구와 버전을 검증한다."""
    
    for label, customer_inquiry, policy_marker in TEST_CASES:
        prompt_spec = builder(customer_inquiry, label)
        
        # 분류 라벨과 고객 질문이 출력 객체에 그대로 전달되는지 확인
        assert prompt_spec.label == label
        assert prompt_spec.customer_inquiry == customer_inquiry
        
        # 해당 라벨의 정책과 실험용 프롬프트 버전이 올바른지 확인
        assert policy_marker in prompt_spec.system
        assert prompt_spec.prompt_version == expected_version
        

def main():
    """A 방식과 B 방식 ,프롬프트를 순서대로 검증한다."""
    
    validate_prompt_builder(
        build_label_specific_prompt,
        "label-specific-v1",
    )
    validate_prompt_builder(
        build_common_policy_prompt,
        "common-policy-v1",
    )
    
    print("프롬프트 구조 검증 완료: 6건 통과")
    
    
if __name__ == "__main__":
    main()