from ..pipeline import Pipeline
import timeit
import shutil
import sys
import os
import random
from time import sleep
import numpy as np
import math
import pickle
from sklearn.svm import SVC


class Training:
    def __init__(self, className: str, fileExtension: str, inputImagePath: str) -> None:
        self.__className = className
        self.__fileExtension = fileExtension
        self.__inputImagePath = inputImagePath

    def exec(self):
        from ..image_augmented import ImageAugmented
        from ..image_preprocessing import ImagePreProcessing

        imagePreProcessingStep = ImagePreprocessing(
            className=self.__className,
            fileExtension=self.__fileExtension,
            inputImagePath=self.__inputImagePath,
        )
        filesCopyStep = FlesCopying()
        faceCrop = FaceCrop()
        modelTraining = ModelTraining()
        modelValidation = ModelValidation()

        pipeline = (
            imagePreProcessingStep
            | filesCopyStep
            | faceCrop
            | modelTraining
            | modelValidation
            | modelValidation
        )
        try:
            for _ in pipeline:
                pass
        except StopIteration:
            return
        except KeyboardInterrupt:
            return
        finally:
            print("[INFO] End of pipeline")


class ImagePreprocessing(Pipeline):
    def __init__(self, className: str, fileExtension: str, inputImagePath: str) -> None:
        self.__isImagePreprocessingOk = False
        self.__isFileCopyingOk = False
        self.__isFaceCropOk = False
        self.__className = className
        self.__fileExtension = fileExtension
        self.__inputImagePath = inputImagePath

    def generator(self):
        data = {
            "isImagePreprocessingOk": self.__isImagePreprocessingOk,
            "isFileCopyingOk": self.__isFileCopyingOk,
            "isFaceCropOk": self.__isFaceCropOk,
            "className": self.__className,
            "fileExtension": self.__fileExtension,
        }
        if self.filter(data):
            yield self.map(data)

    def map(self, data):
        from ..image_augmented import ImageAugmented
        from ..image_preprocessing import ImagePreProcessing

        start = timeit.default_timer()
        fileName = self.__className + self.__fileExtension
        ImagePreProcessing(
            imageURL=self.__inputImagePath, taskName="Denoising", className=self.__className
        ).exec()
        ImageAugmented(imgPath=fileName, saveToDir="preview", class_name="leo").exec()
        stop = timeit.default_timer()
        print(">>> FINISHED on", (stop - start))

        return super().map(data)


class FlesCopying(Pipeline):
    def __init__(self) -> None:
        self.__sourceFolder = "preview/"
        self.__destinationFolder = "mtcnn/training_data/raw/"

    def map(self, data):
        start = timeit.default_timer()
        sourcePath = self.__sourceFolder + data["className"]
        destinationPath = self.__destinationFolder + data["className"]
        try:
            shutil.copytree(sourcePath, destinationPath, dirs_exist_ok=False)
        except FileExistsError as err:
            print("[FileExistsError]", err)

        stop = timeit.default_timer()
        print(">>> [COPYING] FINISHED on", (stop - start))

        return super().map(data)


class FaceCrop(Pipeline):
    def __init__(self) -> None:
        self.__imageSize = 160
        self.__margin = 32
        self.__randomOrder = True
        self.__gpuMemoryFraction = 0.25
        self.__inputDir = "mtcnn/training_data/raw"
        self.__outputDir = "mtcnn/training_data/processed"
        self.__detectMultipleFaces = False

    def map(self, data):
        from ...mtcnn.code.align import detect_face
        from ...mtcnn.code import facenet
        import tensorflow as tf

        sleep(random.random())
        output_dir = os.path.expanduser(self.__outputDir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        src_path, _ = os.path.split(os.path.realpath(__file__))
        facenet.store_revision_info(src_path, output_dir, " ".join(sys.argv))
        dataset = facenet.get_dataset(self.__inputDir)
        print("Creating networks and loading parameters")

        with tf.Graph().as_default():
            sess = tf.compat.v1.Session()
            with sess.as_default():
                pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

        minsize = 20
        threshold = [0.6, 0.7, 0.7]
        factor = 0.709

        random_key = np.random.randint(0, high=99999)
        bounding_boxes_filename = os.path.join(output_dir, f"bounding_boxes_{random_key:05d}.txt")

        with open(bounding_boxes_filename, "w") as text_file:
            nrof_images_total = 0
            nrof_successfully_aligned = 0
            if self.__randomOrder:
                random.shuffle(dataset)
            for cls in dataset:
                output_class_dir = os.path.join(output_dir, cls.name)
                if not os.path.exists(output_class_dir):
                    os.makedirs(output_class_dir)
                    if self.__randomOrder:
                        random.shuffle(cls.image_paths)
                for image_path in cls.image_paths:
                    nrof_images_total += 1
                    filename = os.path.splitext(os.path.split(image_path)[1])[0]
                    output_filename = os.path.join(output_class_dir, filename + ".png")
                    print(image_path)
                    if not os.path.exists(output_filename):
                        try:
                            import imageio

                            img = imageio.imread(image_path)
                        except (IOError, ValueError, IndexError) as e:
                            errorMessage = f"{image_path}: {e}"
                            print(errorMessage)
                        else:
                            if img.ndim < 2:
                                print(f'Unable to align "{image_path}"')
                                text_file.write(f"{output_filename}\n")
                                continue
                            if img.ndim == 2:
                                img = facenet.to_rgb(img)
                            img = img[:, :, 0:3]

                            bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
                            nrof_faces = bounding_boxes.shape[0]
                            if nrof_faces > 0:
                                det = bounding_boxes[:, 0:4]
                                det_arr = []
                                img_size = np.asarray(img.shape)[0:2]
                                if nrof_faces > 1:
                                    if self.__detectMultipleFaces:
                                        for i in range(nrof_faces):
                                            det_arr.append(np.squeeze(det[i]))
                                    else:
                                        bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                                        img_center = img_size / 2
                                        offsets = np.vstack(
                                            [
                                                (det[:, 0] + det[:, 2]) / 2 - img_center[1],
                                                (det[:, 1] + det[:, 3]) / 2 - img_center[0],
                                            ]
                                        )
                                        offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                                        index = np.argmax(
                                            bounding_box_size - offset_dist_squared * 2.0
                                        )
                                        det_arr.append(det[index, :])
                                else:
                                    det_arr.append(np.squeeze(det))

                                for i, det in enumerate(det_arr):
                                    det = np.squeeze(det)
                                    bb = np.zeros(4, dtype=np.int32)
                                    bb[0] = np.maximum(det[0] - self.__margin / 2, 0)
                                    bb[1] = np.maximum(det[1] - self.__margin / 2, 0)
                                    bb[2] = np.minimum(det[2] + self.__margin / 2, img_size[1])
                                    bb[3] = np.minimum(det[3] + self.__margin / 2, img_size[0])
                                    cropped = img[bb[1] : bb[3], bb[0] : bb[2], :]
                                    from PIL import Image

                                    cropped = Image.fromarray(cropped)
                                    scaled = cropped.resize((self.__imageSize, self.__imageSize), Image.BILINEAR)
                                    nrof_successfully_aligned += 1
                                    filename_base, file_extension = os.path.splitext(output_filename)
                                    if self.__detectMultipleFaces:
                                        output_filename_n = f"{filename_base}_{i}{file_extension}"
                                    else:
                                        output_filename_n = f"{filename_base}{file_extension}"
                                    imageio.imwrite(output_filename_n, scaled)
                                    text_file.write(
                                        f"{output_filename_n} {bb[0]} {bb[1]} {bb[2]} {bb[3]}\n"
                                    )
                            else:
                                print(f'Unable to align "{image_path}"')
                                text_file.write(f"{output_filename}\n")

        print(f"Total number of images: {nrof_images_total}")
        print(f"Number of successfully aligned images: {nrof_successfully_aligned}")

        return super().map(data)


class ModelTraining(Pipeline):
    def __init__(self) -> None:
        self.__mode = "TRAIN"
        self.__data_dir = "mtcnn/training_data/processed"
        self.__model = "mtcnn/models/20180402-114759.pb"
        self.__classifier_filename = "mtcnn/models/facemodel.pkl"
        self.__batch_size = 90
        self.__use_split_dataset = False
        self.__test_data_dir = ""
        self.__image_size = 160
        self.__seed = 666
        self.__min_nrof_images_per_class = 20
        self.__nrof_train_images_per_class = 10

    def map(self, data):
        from ...mtcnn.code import facenet
        import tensorflow as tf

        with tf.Graph().as_default():
            with tf.compat.v1.Session() as sess:
                np.random.seed(seed=self.__seed)

                if self.__use_split_dataset:
                    dataset_tmp = facenet.get_dataset(self.__data_dir)
                    train_set, test_set = self.split_dataset(
                        dataset_tmp, self.__min_nrof_images_per_class, self.__nrof_train_images_per_class, facenet
                    )
                    if self.__mode == "TRAIN":
                        dataset = train_set
                    elif self.__mode == "CLASSIFY":
                        dataset = test_set
                else:
                    dataset = facenet.get_dataset(self.__data_dir)

                for cls in dataset:
                    assert len(cls.image_paths) > 0, "There must be at least one image for each class"

                paths, labels = facenet.get_image_paths_and_labels(dataset)

                print(f"Number of classes: {len(dataset)}")
                print(f"Number of images: {len(paths)}")

                print("Loading feature extraction model")
                facenet.load_model(self.__model)

                images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
                embedding_size = embeddings.get_shape()[1]

                print("Calculating features for images")
                nrof_images = len(paths)
                nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / self.__batch_size))
                emb_array = np.zeros((nrof_images, embedding_size))
                for i in range(nrof_batches_per_epoch):
                    start_index = i * self.__batch_size
                    end_index = min((i + 1) * self.__batch_size, nrof_images)
                    paths_batch = paths[start_index:end_index]
                    images = facenet.load_data(paths_batch, False, False, self.__image_size)
                    feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                    emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)

                classifier_filename_exp = os.path.expanduser(self.__classifier_filename)

                if self.__mode == "TRAIN":
                    print("Training classifier")
                    model = SVC(kernel="linear", probability=True)
                    model.fit(emb_array, labels)

                    class_names = [cls.name.replace("_", " ") for cls in dataset]

                    with open(classifier_filename_exp, "wb") as outfile:
                        pickle.dump((model, class_names), outfile)

                    print(f'Saved classifier model to file "{classifier_filename_exp}"')

                elif self.__mode == "CLASSIFY":
                    print("Testing classifier")
                    with open(classifier_filename_exp, "rb") as infile:
                        (model, class_names) = pickle.load(infile)

                    print(f'Loaded classifier model from file "{classifier_filename_exp}"')

                    predictions = model.predict_proba(emb_array)
                    best_class_indices = np.argmax(predictions, axis=1)
                    best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

                    for i in range(len(best_class_indices)):
                        print(f"{i:4d}  {class_names[best_class_indices[i]]}: {best_class_probabilities[i]:.3f}")

                    accuracy = np.mean(np.equal(best_class_indices, labels))
                    print(f"Accuracy: {accuracy:.3f}")

        return super().map(data)

    def split_dataset(dataset, min_nrof_images_per_class, nrof_train_images_per_class, facenet_module):
        train_set = []
        test_set = []
        for cls in dataset:
            paths = cls.image_paths
            if len(paths) >= min_nrof_images_per_class:
                np.random.shuffle(paths)
                train_set.append(facenet_module.ImageClass(cls.name, paths[:nrof_train_images_per_class]))
                test_set.append(facenet_module.ImageClass(cls.name, paths[nrof_train_images_per_class:]))
        return train_set, test_set


class ModelValidation(Pipeline):
    def __init__(self) -> None:
        self.__mode = "CLASSIFY"
        self.__use_split_dataset = False
        self.__data_dir = "mtcnn/training_data/processed"
        self.__classifier_filename = "mtcnn/models/facemodel.pkl"
        self.__min_nrof_images_per_class = 20
        self.__nrof_train_images_per_class = 10
        self.__batch_size = 90
        self.__image_size = 160
        self.__model = "mtcnn/models/20180402-114759.pb"
        self.__seed = 666

    def map(self, data):
        from ...mtcnn.code import facenet
        import tensorflow as tf

        with tf.Graph().as_default():
            with tf.compat.v1.Session() as sess:
                np.random.seed(seed=self.__seed)

                if self.__use_split_dataset:
                    dataset_tmp = facenet.get_dataset(self.__data_dir)
                    train_set, test_set = self.split_dataset(
                        dataset_tmp, self.__min_nrof_images_per_class, self.__nrof_train_images_per_class, facenet
                    )
                    dataset = test_set
                else:
                    dataset = facenet.get_dataset(self.__data_dir)

                for cls in dataset:
                    assert len(cls.image_paths) > 0, "There must be at least one image for each class"

                paths, labels = facenet.get_image_paths_and_labels(dataset)

                print(f"Number of classes: {len(dataset)}")
                print(f"Number of images: {len(paths)}")

                print("Loading feature extraction model")
                facenet.load_model(self.__model)

                images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
                embedding_size = embeddings.get_shape()[1]

                print("Calculating features for images")
                nrof_images = len(paths)
                nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / self.__batch_size))
                emb_array = np.zeros((nrof_images, embedding_size))
                for i in range(nrof_batches_per_epoch):
                    start_index = i * self.__batch_size
                    end_index = min((i + 1) * self.__batch_size, nrof_images)
                    paths_batch = paths[start_index:end_index]
                    images = facenet.load_data(paths_batch, False, False, self.__image_size)
                    feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                    emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)

                classifier_filename_exp = os.path.expanduser(self.__classifier_filename)

                if self.__mode == "TRAIN":
                    print("Training classifier")
                    model = SVC(kernel="linear", probability=True)
                    model.fit(emb_array, labels)

                    class_names = [cls.name.replace("_", " ") for cls in dataset]

                    with open(classifier_filename_exp, "wb") as outfile:
                        pickle.dump((model, class_names), outfile)

                    print(f'Saved classifier model to file "{classifier_filename_exp}"')

                elif self.__mode == "CLASSIFY":
                    print("Testing classifier")
                    with open(classifier_filename_exp, "rb") as infile:
                        (model, class_names) = pickle.load(infile)

                    print(f'Loaded classifier model from file "{classifier_filename_exp}"')

                    predictions = model.predict_proba(emb_array)
                    best_class_indices = np.argmax(predictions, axis=1)
                    best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

                    for i in range(len(best_class_indices)):
                        print(f"{i:4d}  {class_names[best_class_indices[i]]}: {best_class_probabilities[i]:.3f}")

                    accuracy = np.mean(np.equal(best_class_indices, labels))
                    print(f"Accuracy: {accuracy:.3f}")

        return super().map(data)

    def split_dataset(dataset, min_nrof_images_per_class, nrof_train_images_per_class, facenet_module):
        train_set = []
        test_set = []
        for cls in dataset:
            paths = cls.image_paths
            if len(paths) >= min_nrof_images_per_class:
                np.random.shuffle(paths)
                train_set.append(facenet_module.ImageClass(cls.name, paths[:nrof_train_images_per_class]))
                test_set.append(facenet_module.ImageClass(cls.name, paths[nrof_train_images_per_class:]))
        return train_set, test_set
