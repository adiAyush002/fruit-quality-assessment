# Dataset folder

Put your images here. Each of the three tasks has its own folder:

```text
dataset/
├── fruit/        apple, banana, mango, orange
├── ripeness/     unripe, ripe, overripe
└── defect/       healthy, bruised, spotted, rotten
```

Inside every task folder use `train/`, `validation/`, `test/`, and inside each of those one folder per class:

```text
dataset/fruit/train/apple/img001.jpg
dataset/fruit/validation/apple/img240.jpg
dataset/fruit/test/apple/img301.jpg
```

The folder name IS the label. See the main README for details, public dataset suggestions and the helper script.
