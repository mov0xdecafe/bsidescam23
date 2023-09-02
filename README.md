# Identifying Function Boundaries using CNN's

## Quickstart
```sh
git clone [url]
cd bsidescam23
./scripts/build.sh
./scripts/run.sh
cd app
python3 src/driver.py
```

## Results

```
Function Boundary Identification
100%|████████████████████████████████████████████████████████████████████████████████████| 1218/1218 [03:33<00:00,  5.70it/s]
[*] Training Model
100%|██████████████████████████████████████████████████████████████████████████████| 128333/128333 [1:04:55<00:00, 32.94it/s]
[+] Model Trained
[*] Testing Model
100%|█████████████████████████████████████████████████████████████████████████████████| 14260/14260 [01:22<00:00, 173.70it/s]
accuracy: 0.9999357172086651
precision: 0.9931984298863021
recall: 0.9763164022220004
f1-score: 0.9846850624842742
```