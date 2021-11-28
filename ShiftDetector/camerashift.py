import base64
import multiprocessing as mp
import time
import traceback
from typing import Optional, Tuple
import sys
#import cv2
import ffmpeg
import numpy as np
from confa.reports_producer_producer import *
# import redis



def ffmpeg_test(in_file, batch_size):
    probe = ffmpeg.probe(in_file)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    width = int(video_info['width'])
    height = int(video_info['height'])
    cl_channels = 3
    packet_size = width * height * cl_channels

    process = ffmpeg \
        .input(in_file, rtsp_transport='tcp') \
        .output('pipe:', format='rawvideo', pix_fmt='bgr24') \
        .run_async(pipe_stdout=True)

    if batch_size == 1:
        for c in range(1000):
            packet = process.stdout.read(packet_size)
            if not packet:
                break
            res = np.frombuffer(packet, np.uint8).reshape([batch_size, height, width, cl_channels])
    else:
        batch = []
        for c in range(1000):
            packet = process.stdout.read(packet_size)
            if not packet:
                break
            frame = np.frombuffer(packet, np.uint8).reshape([height, width, cl_channels])
            batch.append(frame)
            if len(batch) == batch_size:
                res = np.asarray(batch)
                batch = []
        if batch:
            res = np.asarray(batch)


def ffmpeg_decode_exc(in_file: str, e):
    e_str = e.stderr.decode('utf-8')
    e_msg_idx = e_str.index(in_file)
    return e_str[e_msg_idx:].strip()


def get_ffmpeg_probe(in_file: str, timeout: int = 5 * 60, **kwargs) -> \
        Tuple[Optional[dict], Optional[np.ndarray], Optional[int]]:
    
    print(in_file)
    time_start = time.time()
    cl_channels = 3
    print(f"trying to probe {in_file}")
    while time.time() < time_start + timeout:
        try:
            probe = ffmpeg.probe(in_file, **kwargs)
            video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            width = int(video_info['width'])
            height = int(video_info['height'])
            r_frame_rate = video_info['r_frame_rate']
            r_frame_rate = int(eval(r_frame_rate))
            if width > 0 and height > 0:
                print(f"probed {in_file}")
                return probe, np.asarray([height, width, cl_channels], dtype=int), r_frame_rate
        except ffmpeg.Error as e:
            print(f"failed to probe {in_file}: {ffmpeg_decode_exc(in_file, e)}, retrying")
        except Exception as e:
            print(f"failed to probe {in_file}: {e}, retrying")
    print(f"couldn't probe {in_file} in {timeout} seconds, exiting")
    return None, None, None


def ffmpeg_generator_unbatched(in_file: str, retries=10, debug=False):
    process = None
    try:
        input_kwargs = {}
        if in_file.startswith('rtsp://'):
            input_kwargs['rtsp_transport'] = 'tcp'
        process, shape = ffmpeg_connect(in_file, max(1, retries))
        if process is None:
            return
        packet_size = np.prod(shape)
        while True:
            if debug: print("read packet")
            packet = process.stdout.read(packet_size)
            time_now = time.time()
            if not packet:
                print("no packet")
                process.kill()
                process, shape = ffmpeg_connect(in_file, retries)
                if process is None:
                    break
            frame = np.frombuffer(packet, np.uint8).reshape(shape)
            yield frame, time_now
    except:
        if process is not None:
            process.kill()
        traceback.print_exc()


def redis_generator(camera_id, batch_size=10, timeout=60):
    connect = redis.Redis(
        host=config.REDIS_HOST,
        port=config.REDIS_PORT,
        health_check_interval=120,
        socket_keepalive=True,
    )
    pubsub = connect.pubsub()
    pubsub.subscribe(str(camera_id))
    last_get = time.time()
    if batch_size == 1:
        while True:
            msg = pubsub.get_message()
            if msg:
                print(msg['type'])
                if msg['type'] == 'message':
                    frame, timestamp = decode_from_redis(msg['data'])
                    yield np.asarray([frame.copy()]), [timestamp]
                last_get = time.time()
            if last_get + timeout < time.time():
                break
    else:
        frame_buffer, ts_buffer = [], []
        while True:
            msg = pubsub.get_message()
            if msg:
                if msg['type'] == 'message':
                    frame, timestamp = decode_from_redis(msg['data'])
                    frame_buffer.append(frame.copy())
                    ts_buffer.append(timestamp)
                    if len(frame_buffer) == batch_size:
                        frames = np.asarray(frame_buffer)
                        for i in reversed(range(batch_size)):
                            frame_buffer[i] = None
                            del frame_buffer[i]
                        yield frames, ts_buffer
                        frame_buffer, ts_buffer = [], []
                last_get = time.time()
            if last_get + timeout < time.time():
                if frame_buffer:
                    frames = np.asarray(frame_buffer)
                    for i in reversed(range(len(frame_buffer))):
                        frame_buffer[i] = None
                        del frame_buffer[i]
                    yield frames, ts_buffer
                break


def ffmpeg_generator(in_file: str, batch_size: int = 1, retries=10, debug=False) -> [np.ndarray, list]:
    process = None
    try:
        input_kwargs = {}
        if in_file.startswith('rtsp://'):
            input_kwargs['rtsp_transport'] = 'tcp'
        process, shape = ffmpeg_connect(in_file, max(1, retries))
        if process is None:
            return
        packet_size = np.prod(shape)
        if batch_size == 1:
            while True:
                if debug: print("read packet")
                packet = process.stdout.read(packet_size)
                if not packet:
                    if debug: print("no packet")
                    process.kill()
                    process, shape = ffmpeg_connect(in_file, retries)
                    if process is None:
                        break
                frame = np.frombuffer(packet, np.uint8).reshape(shape)
                yield np.asarray([frame.copy()]), [time.time()]
        else:
            frame_buffer, ts_buffer = [], []
            while True:
                if debug: print("read packet")
                packet = process.stdout.read(packet_size)
                if not packet:
                    if debug: print("no packet")
                    process.kill()
                    process, shape = ffmpeg_connect(in_file, retries)
                    if process is None:
                        if frame_buffer:
                            frames = np.asarray(frame_buffer)
                            for i in reversed(range(len(frame_buffer))):
                                frame_buffer[i] = None
                                del frame_buffer[i]
                            yield frames, ts_buffer
                        break
                if len(frame_buffer) == batch_size:
                    frames = np.asarray(frame_buffer)
                    for i in reversed(range(batch_size)):
                        frame_buffer[i] = None
                        del frame_buffer[i]
                    yield frames, ts_buffer
                    frame_buffer, ts_buffer = [], []
                try:
                    frame = np.frombuffer(packet, np.uint8).reshape(shape)
                    frame_buffer.append(frame.copy())
                    ts_buffer.append(time.time())
                except ValueError:
                    # битый пакет, читаем следующий
                    print(f"bad packet: got {len(packet)} expected {packet_size}")
                    pass

    except:
        if process is not None:
            process.kill()
        traceback.print_exc()


def ffmpeg_connect(in_file, retries, **kwargs):
    ok = False
    for i in range(retries):
        try:
            print(f"trying to connect ({i}): {in_file}")
            probe, shape, frame_rate = get_ffmpeg_probe(in_file)
            if probe is not None:
                ok = True
                print("probe is not None, return process")
                break
        except ffmpeg.Error as e:
            print(f"{ffmpeg_decode_exc(in_file, e)}, retrying")
        except (TypeError, StopIteration):
            # bad probe, retry
            continue
    if not ok:
        print("probe is None")
        return None, None
    process = ffmpeg \
        .input(in_file, **kwargs) \
        .output('pipe:', format='rawvideo', pix_fmt='bgr24') \
        .run_async(pipe_stdout=True)
    return process, shape




def ridoffbest(image):
  iy = np.where(image > 10, 255,0)
  indices = np.where(iy==0)
  iy[indices[0], indices[1], :] = [0, 0, 0]
  return iy


# def detect(gen, path = 'rtps://wow.mov'):
#   read = True
#   ind = 0
#   res = []
#   size = 4
#   otveti = []
#   while read:
#     read = False
#     try:
#       frame1,frame2 = next(gen),next(gen)
#       timer1,timer2 = int(frame1[1][0]),int(frame1[1][0])
#       im1,im2 = frame1[0][0],frame2[0][0]
#       read = True
#       ind+=1
#       ff = im2 - im1
#       fx = ridoffbest(ff)
#       a = np.mean(fx)
      

#       if a > 120:
#         res.append(ind)
        
#         if len(res)>size:
#           res.pop(0)

#         if len(res) == size:
#           divs = []
#           for s in range(len(res)-1):
#             divs.append(abs(res[s+1]-res[s]))

#           o = all([ True if z <= 2 else False for z in divs ])

#           # print('-----------')
#           # print(res)
#           # print(o)
#           if o:
#             otveti = {
#                 'event_type': 'shift_detection',
#                 'rtsp': path,
#                 'time': timer2}
            
#             send_message('monitoring', otveti)
#             print('vse okey')

      
#     except:
#       print('oops',read)
    
    
#   return otveti
    


def detect(gen, path = 'rtps://wow.mov'):
  read = True
  ind = 0
  res = []
  size = 4
  otveti = []
  while read:
    read = False
    try:
      frame1,frame2 = next(gen),next(gen)
      timer1,timer2 = int(frame1[1][0]),int(frame1[1][0])
      im1,im2 = frame1[0][0],frame2[0][0]
      read = True
      ind+=1
      ff = im2 - im1
      fx = ridoffbest(ff)
      a = np.mean(fx)
      

      if a > 120:
        res.append(ind)
#        print(res)
        if len(res) >= size:
          divs = []
          for s in range(len(res)-1):
            divs.append(abs(res[s+1]-res[s]))

          o = all([ True if z <= 2 else False for z in divs ])

          if o:
            # print('-----------')
            # print(res)
            otveti = res
          else:
            res.pop(0)

      if len(otveti) >= size and abs(otveti[-1] - ind) > 2:
        event = {
                'event_type': 'shift_detection',
                'rtsp': path,
                'time': timer2}
            
        send_message('monitoring', event)
        print('message delivered!')
        res = []
        otveti = []   
           
    except:
      print('oops',read)
    
    
  return otveti


if __name__ == "__main__":
    # in_filename = "rtsp://freja.hiof.no:1935/rtplive/_definst_/hessdalen02.stream"
    # in_filename = 'rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov'
    

    in_filename = sys.argv[1]
    gen = ffmpeg_generator(in_filename)
    la_finale = detect(gen,in_filename)




