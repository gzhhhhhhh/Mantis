import argparse
import os
import pickle
import random


DEFAULT_DATA_ROOT = os.environ.get('MAMBA3D_DATA_ROOT', 'data')


def load_processed_modelnet(data_root):
    root = os.path.join(data_root, 'ModelNet', 'modelnet40_normal_resampled')
    train_data_path = os.path.join(root, 'modelnet40_train_8192pts_fps.dat')
    test_data_path = os.path.join(root, 'modelnet40_test_8192pts_fps.dat')

    with open(train_data_path, 'rb') as file_obj:
        train_points, train_labels = pickle.load(file_obj)
    with open(test_data_path, 'rb') as file_obj:
        test_points, test_labels = pickle.load(file_obj)
    return train_points, train_labels, test_points, test_labels


def build_class_index(points_list, labels_list):
    class_index = {}
    for points, label in zip(points_list, labels_list):
        class_id = int(label[0])
        class_index.setdefault(class_id, []).append(points)
    return class_index


def generate_fewshot_split(train_cls_dataset, test_cls_dataset, way, shot, eval_sample, rng):
    train_dataset = []
    test_dataset = []

    keys = list(train_cls_dataset.keys())
    rng.shuffle(keys)

    for episodic_label, original_label in enumerate(keys[:way]):
        train_data_list = list(train_cls_dataset[original_label])
        test_data_list = list(test_cls_dataset[original_label])
        rng.shuffle(train_data_list)
        rng.shuffle(test_data_list)

        if len(train_data_list) <= shot:
            raise ValueError(f'class {original_label} does not have enough training samples for {shot}-shot')
        if len(test_data_list) < eval_sample:
            raise ValueError(f'class {original_label} does not have enough test samples for {eval_sample} eval samples')

        for data in train_data_list[:shot]:
            train_dataset.append((data, episodic_label, original_label))
        for data in test_data_list[:eval_sample]:
            test_dataset.append((data, episodic_label, original_label))

    rng.shuffle(train_dataset)
    rng.shuffle(test_dataset)
    return {
        'train': train_dataset,
        'test': test_dataset,
    }


def main():
    parser = argparse.ArgumentParser(description='Generate ModelNet40 few-shot splits from processed Point-MAE style caches.')
    parser.add_argument('--data-root', default=DEFAULT_DATA_ROOT, help='Root directory containing ModelNet/ and ModelNetFewshot/.')
    parser.add_argument('--output-root', default=None, help='Output directory for generated few-shot splits. Defaults to <data-root>/ModelNetFewshot.')
    parser.add_argument('--ways', nargs='+', type=int, default=[5, 10], help='Episode class counts to generate.')
    parser.add_argument('--shots', nargs='+', type=int, default=[10, 20], help='Support shots to generate.')
    parser.add_argument('--folds', type=int, default=10, help='Number of split files to generate for each setting.')
    parser.add_argument('--eval-sample', type=int, default=20, help='Number of query samples per class.')
    parser.add_argument('--seed', type=int, default=0, help='Base random seed.')
    args = parser.parse_args()

    output_root = args.output_root or os.path.join(args.data_root, 'ModelNetFewshot')
    os.makedirs(output_root, exist_ok=True)

    train_points, train_labels, test_points, test_labels = load_processed_modelnet(args.data_root)
    train_cls_dataset = build_class_index(train_points, train_labels)
    test_cls_dataset = build_class_index(test_points, test_labels)

    for way in args.ways:
        for shot in args.shots:
            save_dir = os.path.join(output_root, f'{way}way_{shot}shot')
            os.makedirs(save_dir, exist_ok=True)

            for fold_idx in range(args.folds):
                rng = random.Random(args.seed + fold_idx + way * 100 + shot * 1000)
                dataset = generate_fewshot_split(
                    train_cls_dataset=train_cls_dataset,
                    test_cls_dataset=test_cls_dataset,
                    way=way,
                    shot=shot,
                    eval_sample=args.eval_sample,
                    rng=rng,
                )
                save_path = os.path.join(save_dir, f'{fold_idx}.pkl')
                with open(save_path, 'wb') as file_obj:
                    pickle.dump(dataset, file_obj)
                print(f'Saved {save_path}')


if __name__ == '__main__':
    main()
