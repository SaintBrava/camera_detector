import json
import time
import traceback
from typing import Dict, Any, Union

from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

from confa import config
from confa._utils import timestamp_for_filename
# from reports_producer.classes import Monitoring


def send(topic: str, data: Dict[str, Any]) -> None:
    cur_time = timestamp_for_filename()
    with open(config.REPORTS_PATH / f"{topic}_{cur_time}.json", 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)


def flush():
    pass


if config.KAFKA_SEND_DATA_ENABLE:

    try:
        producer = KafkaProducer(
            bootstrap_servers=config.KAFKA_SERVER,
            batch_size=config.KAFKA_BATCH_SIZE,
            linger_ms=config.KAFKA_LINGER_MS,
        )

    except NoBrokersAvailable as e:
        print(e)
        print("failed to create KafkaProducer, using stub send and flush")
    else:
        def send(topic: str, data: Union[str, Dict[str, Any]]) -> None:
            if isinstance(data, dict):
                value: bytes = json.dumps(data).encode('utf-8')
            elif isinstance(data, str):
                value = data.encode('utf-8')
            else:
                value = data
            producer.send(topic, value)


        def flush():
            producer.flush()


        print("created KafkaProducer")
elif config.DEBUG_SEND_DATA_ENABLE:
    import datetime as dt
    from collections import defaultdict
    from db.connection import sqla_connect
    from db.models import Aggregation, BackCamera

    def merge_dicts(ds):
        d_merged = {}
        for d in ds:
            for k, v in d.items():
                if k not in d_merged:
                    d_merged[k] = v
                else:
                    if isinstance(v, list):
                        d_merged[k].extend(v)
                    elif isinstance(v, (set, dict)):
                        d_merged[k].update(v)
                    elif d_merged[k] != v:
                        print(f"replace key={k} {d_merged[k]} {v}")
                        d_merged[k] = v
        return d_merged

    def send(topic: str, data: Union[str, Dict[str, Any]]) -> None:
        if topic == config.KAFKA_TOPIC_AGGREGATION:
            with sqla_connect(config.CON_POSTGRES_CONFIG.dsn) as sess:
                camera = sess.query(BackCamera).filter_by(external_id=data["camera_id"]).first()
                members = defaultdict(list)
                camera_name = data["camera_name"]
                for type_of_event, groups in data["type_of_event"].items():
                    for group in groups:
                        for member in group:
                            member_dict = {k: v for k, v in member.items() if v is not None}
                            member_dict['start'] = dt.datetime.fromtimestamp(member_dict['start'])
                            member_dict['stop'] = dt.datetime.fromtimestamp(member_dict['stop'])
                            member_dict['type_of_event'] = [type_of_event]
                            if member['speed'] is not None:
                                speed = member_dict.pop('speed', None)
                                member_dict['speed_avg'] = speed
                            if member['direction'] is not None:
                                direction = member_dict.pop('direction', {})
                                member_dict['start_direction'] = direction.get('start', None)
                                member_dict['stop_direction'] = direction.get('stop', None)
                            if member['url'] is not None:
                                url = member_dict.pop('url', {})
                                if 'frame_url' not in member_dict:
                                    member_dict['frame_url'] = {}
                                if 'frame' in url:
                                    member_dict['frame_url'][type_of_event] = url['frame']
                                if 'grn' in url:
                                    member_dict['grn_url'] = url['grn']
                            members[member_dict['track_id']].append(member_dict)
                for track_id, dicts in members.items():
                    merged_dict = merge_dicts(dicts)
                    agg = Aggregation(camera_id=camera.id, camera_name=camera_name, **merged_dict)
                    sess.add(agg)
                    sess.commit()
else:
    print("KafkaProducer is disabled, using stub send and flush")


def send_message(topic, data):
    try:
        send(topic=topic, data=data)
        # send(topic=topic, data=data.to_dict(encode_json=True))
    except:
        try:
            send(topic=topic, data=json.loads(data.to_json()))
        except:
            print(topic, data)
            traceback.print_exc()
    flush()


# def send_monitoring(camera_id, status, camera_name, lat, lng):
#     monitor = Monitoring(
#         camera_id=camera_id,
#         timestamp=time.time(),
#         status=status,
#         camera_name=camera_name,
#         lat=lat,
#         lng=lng,
#     )
#     send_message(config.KAFKA_TOPIC_MONITORING, monitor)
