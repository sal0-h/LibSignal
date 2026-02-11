import json

data = json.load(open("data/raw_data/grid4x4/grid4x4_roadnet_red.json"))
for inter in data["intersections"]:
    if not inter["virtual"]:
        phases = inter["trafficLight"]["lightphases"]
        print(f"=== Intersection: {inter['id']} ===")
        for idx, p in enumerate(phases):
            print(f"  idx={idx}  time={p['time']}  links={p['availableRoadLinks']}")
        print()
        break  # just check first one
