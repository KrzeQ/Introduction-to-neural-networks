import pygame as pg
import numpy as np
import perceptron as pr
import random
import matplotlib.pyplot as plt
from button import Button

pg.init()
pg.font.init()
font = pg.font.SysFont('Open Sans', 30)
values = np.zeros((5, 5))
rects = []
perceptrons = []
text_numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
training_inputs = []
numbers = [
    [  # 0
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0]
    ],
    [  # 1
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0]
    ],
    [  # 2
        [0.0, 1.0, 1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0]
    ],
    [  # 3
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0]
    ],
    [  # 4
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 1.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0]
    ],
    [  # 5
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0]
    ],
    [  # 6
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0]
    ],
    [  # 7
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0]
    ],
    [  # 8
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0]
    ],
    [  # 9
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0]
    ],
    [  # 0
        [0.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 0.0]
    ],
    [  # 1
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 1.0, 1.0]
    ],
    [  # 2
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0]
    ],
    [  # 3
        [0.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 0.0]
    ],
    [  # 4
        [0.0, 1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0]
    ],
    [  # 5
        [0.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 0.0]
    ],
    [  # 6
        [0.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 0.0]
    ],
    [  # 7
        [0.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0]
    ],
    [  # 8
        [0.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 0.0]
    ],
    [  # 9
        [0.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 0.0]
    ]
]


def init_perceptrons(selected_training_algorithm="RPLA"):
    for _ in range(10):
        perceptrons.append(pr.Perceptron(5 * 5))

    global training_inputs
    training_inputs = [np.ravel(n) for n in numbers]
    if selected_training_algorithm == "RPLA":
        for i in range(20):
            labels = np.zeros(20)
            labels[i % 10] = 1
            labels[i % 10 + 10] = 1
            print(f"Wagi perceptronu dla liczby: {i % 10}")
            perceptrons[i % 10].train_rpla(training_inputs, labels)
    if selected_training_algorithm == "SPLA":
        for i in range(20):
            labels = np.zeros(20)
            labels[i % 10] = 1
            labels[i % 10 + 10] = 1
            print(f"Wagi perceptronu dla liczby: {i % 10}")
            perceptrons[i % 10].train_spla(training_inputs, labels)
    if selected_training_algorithm == "PLA":
        for i in range(20):
            labels = np.zeros(20)
            labels[i % 10] = 1
            labels[i % 10 + 10] = 1
            print(f"Wagi perceptronu dla liczby: {i % 10}")
            perceptrons[i % 10].train_pla(training_inputs, labels)


def init_rectangles():
    left = 10
    top = 10
    width = 40
    height = 40

    rows, cols = values.shape
    for row in range(rows):
        rects_tmp = []
        for value in range(cols):
            rect = pg.Rect(left, top, width, height)
            rects_tmp.append(rect)
            left = left + 50
        left = 10
        top = top + 50
        rects.append(rects_tmp)


def init_buttons(buttons):
    number = 1
    for i in range(3):
        row = []
        x = 10
        y = 290 + i * 30
        for j in range(3):
            row.append(Button(x, y, command=str(number)))
            row.append(Button(x + 40, y, command=str(number), variation=2))
            x += 80
            number += 1
        buttons.append(row)
    row = [Button(10, 380, command='0'), Button(50, 380, command='0', variation=2),
           Button(90, 380, 70, command='clear'),
           Button(170, 380, 70, command='random')]
    buttons.append(row)
    row = [Button(10, 410, 70, command='negation'), Button(90, 410, 70, command='train'),
           Button(170, 410, 70, command='plot')]
    buttons.append(row)


def draw_perceptrons(screen):
    text = "Wykryte cyfry:"
    detected = False
    for x in range(10):
        if perceptrons[x].output(np.ravel(values)) == 1:
            text += f" {x},"
            detected = True

    if not detected:
        text += " Brak."

    textsurface = font.render(text[:-1], False, (0, 0, 0))
    screen.blit(textsurface, (10, 260))
    pg.display.flip()


def draw_rectangles(screen):
    left = 10
    top = 10
    width = 40
    height = 40

    rows, cols = values.shape
    for row in range(rows):
        for value in range(cols):
            rect = pg.Rect(left, top, width, height)
            rects[row][value] = rect
            if values[row][value] == 0:
                pg.draw.rect(screen, (224, 213, 58), rect)
            else:
                pg.draw.rect(screen, (58, 72, 224), rect)
            left = left + 50
        left = 10
        top = top + 50

    pg.display.flip()


def draw_buttons(buttons, screen):
    for row in buttons:
        for button in row:
            pg.draw.rect(screen, (200, 200, 200), (button.x, button.y, button.width, button.height), 0)
            text = font.render(button.command, 1, (0, 0, 0))
            x = int(button.x + button.width/2 - text.get_width()/2)
            y = int(button.y + button.height/2 - text.get_height()/2)
            screen.blit(text, (x, y))

    pg.display.flip()


def main():
    # wybór algorytmu uczącego
    global values
    starting_screen = pg.display.set_mode([310, 100])
    starting_screen.fill((240, 240, 240))
    buttons = [[Button(10, 10, 90, command="SPLA"), Button(110, 10, 90, command="PLA"),
               Button(210, 10, 90, command="RPLA")]]
    selection = True
    running = True
    draw_buttons(buttons, starting_screen)
    selected_algorithm = None
    while selection:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                selection = False
                running = False
            elif event.type == pg.MOUSEBUTTONDOWN:
                x, y = pg.mouse.get_pos()
                for row in buttons:
                    for button in row:
                        if button.clicked(x, y):
                            selected_algorithm = button.command
                            init_perceptrons(button.command)
                            selection = False
    screen = None
    if running:
        screen = pg.display.set_mode([270, 450])
        screen.fill((240, 240, 240))
        buttons = []
        init_buttons(buttons)
        draw_buttons(buttons, screen)
        init_rectangles()
        draw_rectangles(screen)
    # set_perceptrons()

    while running:

        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            elif event.type == pg.MOUSEBUTTONDOWN:
                x, y = pg.mouse.get_pos()
                # kwadraty
                for row in range(len(rects)):
                    for col in range(len(rects[row])):
                        if rects[row][col].collidepoint((x, y)):
                            values[row][col] = values[row][col] * (-1) + 1

                # przyciski
                for row in buttons:
                    for button in row:
                        if button.clicked(x, y):
                            if button.command in text_numbers:
                                if button.variation == 1:
                                    values = np.copy(numbers[int(button.command)])
                                else:
                                    values = np.copy(numbers[int(button.command) + 10])
                            if button.command == "clear":
                                values = np.zeros((5, 5))
                            if button.command == "random":
                                for i in range(len(values)):
                                    for j in range(len(values[i])):
                                        rand = random.randrange(20)
                                        if rand < 1:
                                            values[i][j] = values[i][j] * (-1) + 1
                            if button.command == "negation":
                                values = values * (-1) + 1
                            if button.command == "plot":
                                for i in range(len(perceptrons)):
                                    plt.imshow(np.reshape(perceptrons[i].weights[1:], (5, 5)))
                                    plt.savefig(f"{i}.png")
                            if button.command == "train":
                                if selected_algorithm == "PLA":
                                    for i in range(20):
                                        labels = np.zeros(20)
                                        labels[i % 10] = 1
                                        labels[i % 10 + 10] = 1
                                        print(f"Wagi perceptronu dla liczby: {i % 10}")
                                        perceptrons[i % 10].iterations = 500
                                        perceptrons[i % 10].train_pla(training_inputs, labels)
                                if selected_algorithm == "RPLA":
                                    for i in range(20):
                                        labels = np.zeros(20)
                                        labels[i % 10] = 1
                                        labels[i % 10 + 10] = 1
                                        print(f"Wagi perceptronu dla liczby: {i % 10}")
                                        perceptrons[i % 10].iterations = 500
                                        perceptrons[i % 10].train_rpla(training_inputs, labels)
                                if selected_algorithm == "SPLA":
                                    for i in range(20):
                                        labels = np.zeros(20)
                                        labels[i % 10] = 1
                                        labels[i % 10 + 10] = 1
                                        print(f"Wagi perceptronu dla liczby: {i % 10}")
                                        perceptrons[i % 10].iterations = 500
                                        perceptrons[i % 10].train_spla(training_inputs, labels)
                screen.fill((240, 240, 240))
                draw_rectangles(screen)
                draw_perceptrons(screen)
                draw_buttons(buttons, screen)

    pg.quit()


if __name__ == "__main__":
    main()
