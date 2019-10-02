```
data
├── multi_source
│   ├── raw
│   │   ├── clipart
│   │   ├── clipart_test.txt
│   │   ├── clipart_train.txt
│   │   ├── infograph
│   │   ├── infograph_test.txt
│   │   ├── infograph_train.txt
│   │   ├── painting
│   │   ├── painting_test.txt
│   │   ├── painting_train.txt
│   │   ├── quickdraw
│   │   ├── quickdraw_test.txt
│   │   ├── quickdraw_train.txt
│   │   ├── real
│   │   ├── real_test.txt
│   │   ├── real_train.txt
│   │   ├── sketch
│   │   ├── sketch_test.txt
│   │   └── sketch_train.txt
│   └── tfrecords
│       ├── clipart_test_000.tfrecord
│       └── ...
└── semi_supervised
    ├── raw
    │   ├── clipart
    │   ├── clipart_labeled.txt
    │   ├── clipart_unlabeled.txt
    │   ├── painting
    │   ├── painting_labeled.txt
    │   ├── painting_unlabeled.txt
    │   ├── sketch
    │   ├── sketch_labeled.txt
    │   └── sketch_unlabeled.txt
    └── tfrecords
        ├── clipart_labeled_000.tfrecord
        └── ...
```