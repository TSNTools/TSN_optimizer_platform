# clean_existing_fattree_instance.py

import json
from pathlib import Path


def clean_instance(
    in_path,
    out_path,
    drop_unscheduled=True,
    drop_out_of_cycle=True,
    renumber=True,
):
    with open(in_path, "r") as f:
        data = json.load(f)

    streams = data.get("streams", [])
    nodes = data.get("nodes", [])

    # stream ids that actually appear in schedules
    scheduled_ids = set()

    # stream ids that have at least one invalid scheduled window
    bad_ids = set()

    # collect scheduled stream ids and detect bad windows
    for node in nodes:
        for port_name, port in node.get("ports", {}).items():
            cycle = port.get("scheduleCycle", None)

            for w in port.get("schedule", []):
                seq = w.get("stream_sequence", [])
                start = int(w.get("start", 0))
                duration = int(w.get("duration", 0))
                end = start + duration

                for sid in seq:
                    sid = int(sid)
                    scheduled_ids.add(sid)

                    if drop_out_of_cycle and cycle is not None:
                        if start < 0 or end > int(cycle):
                            bad_ids.add(sid)

    old_keep_ids = []

    for old_id, stream in enumerate(streams):
        if drop_unscheduled and old_id not in scheduled_ids:
            continue

        if old_id in bad_ids:
            continue

        # Also check stream interval consistency
        interval = int(stream.get("interval", 0))
        if interval > 0:
            stream_bad = False

            for node in nodes:
                for port in node.get("ports", {}).values():
                    for w in port.get("schedule", []):
                        seq = [int(x) for x in w.get("stream_sequence", [])]
                        if old_id not in seq:
                            continue

                        start = int(w.get("start", 0))
                        end = start + int(w.get("duration", 0))

                        if start < 0 or end > interval:
                            stream_bad = True
                            break

                    if stream_bad:
                        break
                if stream_bad:
                    break

            if stream_bad:
                continue

        old_keep_ids.append(old_id)

    keep_set = set(old_keep_ids)

    if renumber:
        id_map = {old_id: new_id for new_id, old_id in enumerate(old_keep_ids)}
    else:
        id_map = {old_id: old_id for old_id in old_keep_ids}

    # clean and renumber streams
    new_streams = []

    for old_id in old_keep_ids:
        s = dict(streams[old_id])
        new_id = id_map[old_id]
        s["name"] = f"stream-{new_id}"
        s["originalStreamId"] = new_id
        new_streams.append(s)

    # clean schedules
    for node in nodes:
        for port_name, port in node.get("ports", {}).items():
            old_sched = port.get("schedule", [])
            new_sched = []

            for w in old_sched:
                old_seq = [int(x) for x in w.get("stream_sequence", [])]
                kept_seq = [id_map[x] for x in old_seq if x in keep_set]

                if not kept_seq:
                    continue

                nw = dict(w)
                nw["stream_sequence"] = kept_seq
                new_sched.append(nw)

            if new_sched:
                port["schedule"] = new_sched
            else:
                port.pop("schedule", None)
                port.pop("basetime", None)
                port.pop("scheduleCycle", None)

    data["streams"] = new_streams

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    print("Input streams:", len(streams))
    print("Scheduled ids:", len(scheduled_ids))
    print("Bad out-of-cycle ids:", len(bad_ids))
    print("Kept streams:", len(new_streams))
    print("Removed streams:", len(streams) - len(new_streams))
    print("Saved:", out_path)


if __name__ == "__main__":
    clean_instance(
        in_path="instances/instance2.json",
        out_path="instance2_clean_initial.json",
    )