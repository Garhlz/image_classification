import timm

# 1. 检查所有可用的efficientnet模型
available_models = timm.list_models('*efficientnet*')
print("可用的EfficientNet模型:")
for model in available_models:
    print(f"- {model}")

# 2. 检查特定的模型是否可用
if 'tf_efficientnetv2_s_in21ft1k' in available_models:
    print("\n找到目标模型!")
else:
    print("\n目标模型不在默认列表中")

# 3. 检查是否可以创建模型
try:
    model = timm.create_model('tf_efficientnetv2_s_in21ft1k', pretrained=False)
    print(model)
    print("\n可以创建模型")
except Exception as e:
    print(f"\n无法创建模型: {e}")