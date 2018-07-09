import os
import cv2
import sys
import time
import numpy as np

ibug_face_tracker_path = 'D:\\hhj\\dlib_and_chehra_stuff'   # You'll need to change this
sys.path.append(ibug_face_tracker_path)
import ibug_face_tracker


def landmark_localiser(job_queue, result_queue):
    # Initialise the tracker
    ert_model_path = os.path.join(ibug_face_tracker_path, 'models',
                                  'new3_68_pts_UAD_1_tr_6_cas_15.dat')
    svr_model_path = os.path.join(ibug_face_tracker_path, 'models',
                                  'additional_svrs.model')
    tracker = ibug_face_tracker.FaceTracker(ert_model_path, svr_model_path)
    tracker.soft_failure_threshold = -1e6
    tracker.hard_failure_threshold = -1e6

    # Localise the facial landmarks in the frames
    while True:
        job = job_queue.get()
        if job is None:
            job_queue.put(None)
            break
        else:
            try:
                tracker.track(cv2.imread(job['image_path']), job['face_box'])
                result = dict()
                result['index'] = job['index']
                result['ibug_facial_landmarks'] = tracker.facial_landmarks
                result['ibug_eye_landmarks'] = tracker.eye_landmarks
                result['ibug_head_pose'] = (tracker.pitch, tracker.yaw, tracker.roll)
                result['ibug_fitting_scores'] = tracker.most_recent_fitting_scores
                result_queue.put(result)
            except:
                pass


def landmark_organiser(result_queue, result_list):
    number_of_jobs = len(result_list)
    last_check_time = time.time()
    processed_images = 0
    while True:
        result = result_queue.get()
        if result is None:
            result_queue.put(None)
            break
        else:
            idx = result['index']
            result.pop('index')
            result_list[idx] = result
            processed_images = processed_images + 1
            current_time = time.time()
            if current_time - last_check_time > 30.0:
                print('%d of %d images processed.' % (processed_images, number_of_jobs))
                last_check_time = current_time


def extract_face_image(image, landmarks, target_size=(60, 60), margin=(0.1, 0.1, 0.1, 0.1)):
    # Ignore the chin points
    if landmarks.shape[0] == 68:
        landmarks = landmarks[17:]

    # Find the face box
    top_left = np.min(landmarks, axis=0)
    bottom_right = np.max(landmarks, axis=0)
    face_size = bottom_right - top_left
    face_box = [int(np.floor(top_left[0] - face_size[0] * margin[0])),
                int(np.floor(top_left[1] - face_size[1] * margin[1])),
                int(np.ceil(bottom_right[0] + face_size[0] * margin[2])) + 1,
                int(np.ceil(bottom_right[1] + face_size[1] * margin[3])) + 1]

    # Make the face box square
    difference = (face_box[3] - face_box[1]) - (face_box[2] - face_box[0])
    if difference > 0:
        face_box[0] -= difference // 2
        face_box[2] += difference - difference // 2
        pass
    elif difference < 0:
        difference = -difference
        face_box[1] -= difference // 2
        face_box[3] += difference - difference // 2

    # Pad the image when necessary
    padding = np.zeros((image.ndim, 2), dtype=int)
    if face_box[0] < 0:
        padding[1][0] = -face_box[0]
    if face_box[2] > image.shape[1]:
        padding[1][1] = face_box[2] - image.shape[1]
    if face_box[1] < 0:
        padding[0][0] = -face_box[1]
    if face_box[3] > image.shape[0]:
        padding[0][1] = face_box[3] - image.shape[0]
    image = np.pad(image, padding, 'symmetric')
    if face_box[0] < 0:
        face_box[2] -= face_box[0]
        face_box[0] = 0
    if face_box[1] < 0:
        face_box[3] -= face_box[1]
        face_box[1] = 0

    # Extract the face image
    face_image = cv2.resize(image[face_box[1]: face_box[3], face_box[0]: face_box[2]],
                            target_size, interpolation=cv2.INTER_CUBIC)
    return face_image


def face_extractor(job_queue, output_queue, target_size=(60, 60), margin=(0.1, 0.1, 0.1, 0.1)):
    while True:
        job = job_queue.get()
        if job is None:
            job_queue.put(None)
            break
        else:
            try:
                face_image = extract_face_image(cv2.imread(job['image_path']),
                                                job['landmarks'], target_size, margin)
                cv2.imwrite(job['output_path'], face_image)
                output_queue.put(job['output_path'])
            except:
                pass


def face_extractor_monitor(output_queue):
    extract_face_images = 0
    last_check_time = time.time()
    while True:
        output_path = output_queue.get()
        if output_path is None:
            output_queue.put(None)
            break
        else:
            extract_face_images = extract_face_images + 1
            current_time = time.time()
            if current_time - last_check_time > 30.0:
                print('%d face images extracted.' % extract_face_images)
                last_check_time = current_time
