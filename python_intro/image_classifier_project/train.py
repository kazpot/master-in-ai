import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from utils import load_data, build_model, save_checkpoint

def main():
    parser = argparse.ArgumentParser(description="Train a flower classifier.")
    parser.add_argument("data_dir", type=str, help="Path to dataset root (e.g., flowers)")
    parser.add_argument("--save_dir", type=str, default="checkpoint.pth", help="Checkpoint file path")
    parser.add_argument("--arch", type=str, default="vgg16", help='Model architecture: "vgg16" or "vgg13"')
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--hidden_units", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    image_datasets, dataloaders = load_data(args.data_dir, batch_size=args.batch_size)
    output_classes = len(image_datasets["train"].classes)

    model = build_model(arch=args.arch, hidden_units=args.hidden_units, output_classes=output_classes)
    model.to(device)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    steps = 0
    print_every = 50

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in dataloaders["train"]:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            logps = model(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for v_inputs, v_lables in dataloaders["valid"]:
                        v_inputs, v_lables = v_inputs.to(device), v_lables.to(device)
                        v_logps = model(v_inputs)
                        v_loss = criterion(v_logps, v_lables)
                        val_loss += v_loss.item()

                        ps = torch.exp(v_logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        val_correct += (top_class.squeeze(1) == v_lables).sum().item()
                        val_total += v_lables.size(0)

                print(f"Epoch {epoch+1}/{args.epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Valid loss: {val_loss/len(dataloaders['valid']):.3f}.. "
                      f"Valid acc: {val_correct/val_total:.3f}")
                running_loss = 0.0
                model.train()

    print(f"Saved checkpoint: {args.save_dir}")

    model.class_to_idx = image_datasets["train"].class_to_idx
    save_checkpoint(
        filepath=args.save_dir,
        model=model,
        optimizer=optimizer,
        epochs=args.epochs,
        arch=args.arch,
        learning_rate=args.learning_rate,
        hidden_units=args.hidden_units,
        class_to_idx=model.class_to_idx,
    )
    print(f"Saved checkpoint: {args.save_dir}")

# example:
# python train.py flowers --arch vgg16 --epochs 5 --learning_rate 0.001 --hidden_units 1024 --gpu --save_dir checkpoint.pth
if __name__ == "__main__":
    main()