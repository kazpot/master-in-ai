import argparse
import torch
from utils import load_checkpoint, predict, load_category_names

def main():
    parser = argparse.ArgumentParser(description="Predict flower name from an image.")
    parser.add_argument("image_path", type=str, help="Path to input image")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint.pth")
    parser.add_argument("--top_k", type=int, default=5, help="Return top K classes")
    parser.add_argument("--category_names", type=str, default=None, help="Path to cat_to_name.json")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    args = parser.parse_args()

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    model = load_checkpoint(args.checkpoint, device=device)
    probs, classes = predict(args.image_path, model, topk=args.top_k, device=device)

    if args.category_names:
        cat_to_name = load_category_names(args.category_names)
        names = [cat_to_name[c] for c in classes]
    else:
        names = classes

    print(f"Prediction: {names[0]}  (class: {classes[0]})  prob: {probs[0]:.4f}")
    print("Top K:")
    for p, c, n in zip(probs, classes, names):
        print(f"  {p:.4f}  {n}  (class: {c})")

# example command:
# python predict.py flowers/test/1/image_06743.jpg checkpoint.pth --top_k 5 --category_names cat_to_name.json --gpu
if __name__ == "__main__":
    main()