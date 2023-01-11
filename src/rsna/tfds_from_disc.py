from __future__ import print_function, division

import tensorflow as tf
import tf2lib as tl
import glob


def normalize_img(img, special_normalisation):
    if not special_normalisation or special_normalisation == tf.keras.applications.inception_v3.preprocess_input:
        return img / tf.reduce_max(img) * 2 - 1
    elif special_normalisation == tf.keras.applications.densenet.preprocess_input:
        return img / tf.reduce_max(img)


def make_dataset(img_paths, batch_size, load_size, crop_size, training, drop_remainder=True, shuffle=True, repeat=1,
                 labels=None, special_normalisation=None, channels=3):
    """
    Returns a preprocesed batched dataset. If train=True then augmentations are applied.
    """
    if training:
        @tf.function
        def _map_fn(img, label=None):  # preprocessing
            img = tf.cast(img, tf.float32)
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_contrast(img, 0.7, 1.3)
            img = tf.image.random_brightness(img, 0.2)
            """gamma = tf.random.uniform(minval=0.8, maxval=1.2, shape=[1, ])
            img = tf.image.adjust_gamma(img, gamma=gamma[0])"""
            img = tf.image.resize_with_pad(img, load_size, load_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            img = tf.image.random_crop(img, [crop_size, crop_size, tf.shape(img)[-1]])
            img = normalize_img(img, special_normalisation)
            if channels == 1:
                img = tf.image.rgb_to_grayscale(img)
            if label is not None:
                return img, label
            return img
    else:
        @tf.function
        def _map_fn(img, label=None):  # preprocessing
            img = tf.cast(img, tf.float32)
            img = tf.image.resize_with_pad(img, crop_size, crop_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            img = tf.image.random_crop(img, [crop_size, crop_size, tf.shape(img)[-1]])
            img = normalize_img(img, special_normalisation)
            if channels == 1:
                img = tf.image.rgb_to_grayscale(img)
            if label is not None:
                return img, label
            return img

    return tl.disk_image_batch_dataset(img_paths,
                                       batch_size,
                                       drop_remainder=drop_remainder,
                                       map_fn=_map_fn,
                                       shuffle=shuffle,
                                       repeat=repeat,
                                       labels=labels)


def _set_repeat(repeat, A_img_paths, B_img_paths):
    # zip two datasets aligned by the longer one
    if repeat:
        A_repeat = B_repeat = None  # cycle both
    else:
        if len(A_img_paths) >= len(B_img_paths):
            A_repeat = 1
            B_repeat = None  # cycle the shorter one
        else:
            A_repeat = None  # cycle the shorter one
            B_repeat = 1
    return A_repeat, B_repeat


def make_zip_dataset(A_img_paths, B_img_paths, batch_size, load_size, crop_size, training, shuffle=False, repeat=False,
                     special_normalisation=None, channels=3):
    # zip two datasets aligned by the longer one
    A_repeat, B_repeat = _set_repeat(repeat, A_img_paths, B_img_paths)

    A_dataset = make_dataset(A_img_paths, batch_size, load_size, crop_size, training, drop_remainder=True,
                             shuffle=shuffle, repeat=A_repeat, special_normalisation=special_normalisation,
                             channels=channels)
    B_dataset = make_dataset(B_img_paths, batch_size, load_size, crop_size, training, drop_remainder=True,
                             shuffle=shuffle, repeat=B_repeat, special_normalisation=special_normalisation,
                             channels=channels)

    A_B_dataset = tf.data.Dataset.zip((A_dataset, B_dataset))
    len_dataset = max(len(A_img_paths), len(B_img_paths)) // batch_size
    return A_B_dataset, len_dataset


def make_concat_dataset(A_img_paths, B_img_paths, batch_size, load_size, crop_size, training, shuffle=False,
                        special_normalisation=None):
    dataset_length = len(A_img_paths) + len(B_img_paths)
    class_labels = [(1, 0) for _ in range(len(A_img_paths))]
    class_labels.extend([(0, 1) for _ in range(len(B_img_paths))])
    A_img_paths.extend(B_img_paths)  # becoming all_image_paths
    all_image_paths = A_img_paths
    return make_dataset(all_image_paths, batch_size, load_size, crop_size, training, drop_remainder=True,
                        shuffle=shuffle, repeat=1, labels=class_labels,
                        special_normalisation=special_normalisation), dataset_length


def get_rsna_ds_split_class(tfds_path, batch_size, crop_size, load_size, special_normalisation=None, channels=3):
    """
    Method loads the RSNA dataloaders that return two samples: one of class A and one of class B.
    Can be used to train CycleGANs.
    tfds_path: Path to tensorflow datasets directory.
    batch_size: Batch size for the data loader.
    crop_size: Final image size that will be cropped to.
    load_size: The image will be loaded with this size.
    special_normalisation: Can be any normalisation from keras preprocessing (e.g. inception_preprocessing)
    """
    A_train = glob.glob(tfds_path + "/rsna_data/train/normal/normal/*")
    B_train = glob.glob(tfds_path + "/rsna_data/train/abnormal/abnormal/*")
    A_valid = glob.glob(tfds_path + "/rsna_data/validation/normal/normal/*")
    B_valid = glob.glob(tfds_path + "/rsna_data/validation/abnormal/abnormal/*")
    A_test = glob.glob(tfds_path + "/rsna_data/test/normal/normal/*")
    B_test = glob.glob(tfds_path + "/rsna_data/test/abnormal/abnormal/*")

    A_B_dataset, len_dataset_train = make_zip_dataset(A_train, B_train, batch_size, load_size,
                                                      crop_size, training=True, repeat=False,
                                                      special_normalisation=special_normalisation,
                                                      channels=channels)

    A_B_dataset_valid, _ = make_zip_dataset(A_valid, B_valid, batch_size, load_size,
                                            crop_size, training=False, repeat=True,
                                            special_normalisation=special_normalisation,
                                            channels=channels)

    A_B_dataset_test, _ = make_zip_dataset(A_test, B_test, batch_size, load_size,
                                           crop_size, training=False, repeat=True,
                                           special_normalisation=special_normalisation,
                                           channels=channels)

    return A_B_dataset, A_B_dataset_valid, A_B_dataset_test, len_dataset_train


def get_rsna_ds_for_clf(tfds_path, batch_size, crop_size, load_size, special_normalisation=None):
    """
    Method loads the RSNA data. Can be used to train classifiers.
    tfds_path: Path to tensorflow datasets directory.
    batch_size: Batch size for the data loader.
    crop_size: Final image size that will be cropped to.
    load_size: The image will be loaded with this size.
    special_normalisation: Can be any normalisation from keras preprocessing (e.g. inception_preprocessing)
    """
    A_train = glob.glob(tfds_path + "/rsna_data/train/normal/normal/*")
    B_train = glob.glob(tfds_path + "/rsna_data/train/abnormal/abnormal/*")
    A_valid = glob.glob(tfds_path + "/rsna_data/validation/normal/normal/*")
    B_valid = glob.glob(tfds_path + "/rsna_data/validation/abnormal/abnormal/*")
    A_test = glob.glob(tfds_path + "/rsna_data/test/normal/normal/*")
    B_test = glob.glob(tfds_path + "/rsna_data/test/abnormal/abnormal/*")

    A_B_dataset, len_dataset_train = make_concat_dataset(A_train, B_train, batch_size,
                                                         load_size,
                                                         crop_size, training=True, shuffle=True,
                                                         special_normalisation=special_normalisation)

    A_B_dataset_valid, _ = make_concat_dataset(A_valid, B_valid, batch_size,
                                               load_size,
                                               crop_size, training=False,
                                               special_normalisation=special_normalisation)

    A_B_dataset_test, _ = make_concat_dataset(A_test, B_test, batch_size, load_size,
                                              crop_size, training=False,
                                              special_normalisation=special_normalisation)
    return A_B_dataset, A_B_dataset_valid, A_B_dataset_test, len_dataset_train


def get_rsna_TEST_ds_split_class(tfds_path, batch_size, crop_size, load_size, special_normalisation=None, channels=3,
                                 training=False):
    """
    Method loads the TEST RSNA dataloaders that return two samples: one of class A and one of class B.
    Can be used to test CycleGANs.
    tfds_path: Path to tensorflow datasets directory.
    batch_size: Batch size for the data loader.
    crop_size: Final image size that will be cropped to.
    load_size: The image will be loaded with this size.
    special_normalisation: Can be any normalisation from keras preprocessing (e.g. inception_preprocessing)
    """
    A_train = glob.glob(tfds_path + "/rsna_data/train/normal/normal/*")
    B_train = glob.glob(tfds_path + "/rsna_data/train/abnormal/abnormal/*")
    A_valid = glob.glob(tfds_path + "/rsna_data/validation/normal/normal/*")
    B_valid = glob.glob(tfds_path + "/rsna_data/validation/abnormal/abnormal/*")
    # merge train and valid to not exclude images
    A_train.extend(A_valid)
    B_train.extend(B_valid)
    A_test = glob.glob(tfds_path + "/rsna_data/test/normal/normal/*")
    B_test = glob.glob(tfds_path + "/rsna_data/test/abnormal/abnormal/*")

    A_dataset = make_dataset(A_train, batch_size, load_size, crop_size, training=training, drop_remainder=True,
                             shuffle=False, repeat=1, special_normalisation=special_normalisation,
                             channels=channels)
    B_dataset = make_dataset(B_train, batch_size, load_size, crop_size, training=training, drop_remainder=True,
                             shuffle=False, repeat=1, special_normalisation=special_normalisation,
                             channels=channels)

    A_dataset_test = make_dataset(A_test, batch_size, load_size, crop_size, training=training, drop_remainder=True,
                                  shuffle=False, repeat=1, special_normalisation=special_normalisation,
                                  channels=channels)
    B_dataset_test = make_dataset(B_test, batch_size, load_size, crop_size, training=training, drop_remainder=True,
                                  shuffle=False, repeat=1, special_normalisation=special_normalisation,
                                  channels=channels)
    return A_dataset, B_dataset, A_dataset_test, B_dataset_test
