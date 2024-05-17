#!/usr/bin/env python3

import argparse
from timeit import default_timer as timer
from PIL import Image, ImageDraw, ImageFont
from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file

def inference_pycoral(runs, image, model, output, label=None):
    interpreter = common.make_interpreter(model)
    interpreter.allocate_tensors()
    labels = read_label_file(label) if label else None
    img = Image.open(image)
    draw = ImageDraw.Draw(img, 'RGBA')
    helvetica = ImageFont.truetype("./Helvetica.ttf", size=72)
    initial_h, initial_w = img.size
    frame = img.resize((300, 300))

    print("Running inference for", runs, "times.")

    start = timer()
    for _ in range(runs):
        common.set_input(interpreter, frame)
        interpreter.invoke()
        ans = detect.get_objects(interpreter, 0.05, (initial_h / 300), (initial_w / 300))
    end = timer()

    print('Elapsed time is', ((end - start) / runs) * 1000, 'ms')

    if ans:
        print("Processing output")
        for obj in ans:
            if obj.score > 0.5:
                print(labels[obj.id], 'score =', obj.score) if labels else print('score =', obj.score)
                bbox = obj.bbox.flatten().tolist()
                print('box =', bbox)
                bbox = [bbox[1], bbox[0], bbox[3], bbox[2]]  # Convert from (ymin, xmin, ymax, xmax) to (xmin, ymin, xmax, ymax)
                draw_rectangle(draw, bbox, (0, 128, 128, 20), width=5)
                if labels:
                    draw.text((bbox[0] + 20, bbox[1] + 20), labels[obj.id], fill=(255, 255, 255, 20), font=helvetica)
        img.save(output)
        print('Saved to', output)
    else:
        print('No objects detected!')

# Function to draw a rectangle with width > 1
def draw_rectangle(draw, coordinates, color, width=1):
    for i in range(width):
        rect_start = (coordinates[0] - i, coordinates[1] - i)
        rect_end = (coordinates[2] + i, coordinates[3] + i)
        draw.rectangle((rect_start, rect_end), outline=color, fill=color)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Path of the detection model.', required=True)
    parser.add_argument('--label', help='Path of the labels file.')
    parser.add_argument('--input', help='File path of the input image.', required=True)
    parser.add_argument('--output', help='File path of the output image.')
    parser.add_argument('--runs', help='Number of times to run the inference', type=int, default=1)
    args = parser.parse_args()

    if args.output:
        output_file = args.output
    else:
        output_file = 'out.jpg'

    if args.label:
        label_file = args.label
    else:
        label_file = None

    inference_pycoral(args.runs, args.input, args.model, output_file, label_file)

if __name__ == '__main__':
    main()
