from recbole.quick_start import load_data_and_model


def quick_test(model_path):
    """
    快速评估方法
    """
    # 如果模型文件包含配置信息，可以直接使用
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(model_file=model_path)

    # 创建训练器进行评估
    from recbole.trainer import Trainer
    trainer = Trainer(config, model)

    test_result = trainer.evaluate(test_data, load_best_model=False, show_progress=True)

    for metric, value in test_result.items():
        print(f"{metric}: {value}")

    return test_result


# 使用示例
if __name__ == '__main__':
    model = './saved/SASRec-Sep-10-2025_14-43-37.pth'
    results = quick_test(model)