def count_item_in_TAT(file):
    with open(file = file, mode='r') as f:
        items = []
        for line in f:
            item = line.strip().split(' ')[2]
            if item not in items:
                items.append(item)
        item_n = len(items)
        print(item_n)


def count_ui_txt(file):
    with open(file = file, mode='r') as f:
        items = []
        users = []
        for line in f:
            line = line.strip().split(' ')
            user = line[0]
            if user not in users:
                users.append(user)
            item = line[1]
            if item not in items:
                items.append(item)
        item_n = len(items)
        user_n = len(users)
        print(user_n, item_n)
