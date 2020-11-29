class Button(object):
    def __init__(self, x, y, width=30, height=25, command='', variation=1):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.command = command

    def clicked(self, x, y):
        return (self.x < x < self.x+self.width) and (self.y < y < self.y+self.height)