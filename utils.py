import os
import queue
import pickle
import random
import string
import datetime


def track_dir(file_path):
    absolute_path = os.path.abspath(file_path)
    directory = os.path.dirname(absolute_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def random_string(length):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))


def update_log(string, path_file='temp/log.txt'):
    string = f'{datetime.datetime.now()}; {string}'
    print(string)
    track_dir(path_file)
    if not os.path.exists(path_file):
        open(path_file, 'w+', encoding="utf-8")
    with open(path_file, "a", encoding="utf-8") as file:
        file.write(string + '\n')


def random_proportional_resolution(base_width=70):
    aspect_ratios = [
        (12, 8),   # Горизонтальное
        (8, 12),   # Вертикальное
        (16, 9),  # Широкоэкранный
        (9, 16),  # Вертикальный широкоэкранный
        (12, 12),    # Квадратный
        (18, 6),
        (16, 10),
        (10, 12)
    ]

    width_coefficient, height_coefficient = random.choice(aspect_ratios)

    width = int(base_width * width_coefficient)
    height = int(base_width * height_coefficient)

    height = height - (height % 8)
    width = width - (width % 8)

    return [width, height]


def calculate_dimensions(aspect_ratio, max_dimension):

    height = max_dimension
    width = int(height / max(aspect_ratio) * min(aspect_ratio))

    width = (width // 8) * 8
    height = (height // 8) * 8

    if aspect_ratio[0] > aspect_ratio[1]:
        width, height = height, width

    return width, height


def save_queue(queue_instance, filename):
    with open(filename, 'wb') as file:
        pickle.dump(list(queue_instance.queue), file)


def load_queue(filename):
    with open(filename, 'rb') as file:
        return queue.Queue(pickle.load(file))
