import argparse
from train import get_image_batch, get_img
import cv2
from transform import restore_from_checkpoint
# parser = argparse.ArgumentParser(description='Train a Fast Style Network')
# parser.add_argument('integers', metavar='N', type=int, nargs='+',
#                     help='an integer for the accumulator')

def main():
    parser = argparse.ArgumentParser(description='Train a Fast Style Network')
    parser.add_argument('--ckpt', type=str,
                        help='path to checkpoint model to use for styling')
    parser.add_argument('--style', type=str,
                        help='path to style image to use for styling')
    parser.add_argument('--content', type=str,
                        help='path to content image to apply style on')
    parser.add_argument('--save-path', type=str,
                        help='path to where to save the resulting image')
    args = parser.parse_args()
    transform_net, _ = restore_from_checkpoint(args.ckpt)
    contents = get_image_batch([args.content])
    results = transform_net(contents)
    cv2.imwrite(args.save_path, cv2.cvtColor(results.numpy()[0], cv2.COLOR_RGB2BGR));
if __name__ == "__main__":
    main()
