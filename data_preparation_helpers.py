import os
import cv2
import sys
import time

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
