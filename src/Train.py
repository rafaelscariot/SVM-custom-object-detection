import dlib


class TrainSVM:

    def train(self):
        options = dlib.simple_object_detector_training_options()
        options.add_left_right_image_flips = True
        options.C = 5

        dlib.train_simple_object_detector('../resources/model.xml', '../resources/model.svm', options)


if __name__ == '__main__':
    TrainSVM().train()