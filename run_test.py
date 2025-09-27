from recbole.quick_start import load_data_and_model


def test_previous(model_path):
    """
        This function can only test with previous settings
    """

    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(model_file=model_path)

    from recbole.trainer import Trainer
    trainer = Trainer(config, model)

    test_result = trainer.evaluate(test_data, load_best_model=False, show_progress=True)

    for metric, value in test_result.items():
        print(f"{metric}: {value}")

    return test_result


if __name__ == '__main__':
    path = './saved/SASRec-Sep-10-2025_14-43-37.pth'
    results = test_previous(model_path=path)
