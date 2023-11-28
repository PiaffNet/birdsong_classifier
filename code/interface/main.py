def preprocess():
    print('Preprocessing...')
    return 1
def train():
    print('Training...')
    return 1


if __name__ == '__main__':
    try:
        preprocess()
        train()
    except:
        print("Error")
