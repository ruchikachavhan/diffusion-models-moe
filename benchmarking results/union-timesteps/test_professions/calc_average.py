# read results.json which is a list of dictionaries and calculate average of all keys
import json
import numpy as np

def main():
    with open('results.json', 'r') as f:
        results = json.load(f)
        print("Results: ", results)
        keys = results[0].keys()
        print("Keys: ", keys)
        mean_results = {}
        for key in keys:
            mean_results[key] = []
            for result in results:
                mean_results[key].append(result[key] if result[key] > 0.45 else 1)
            mean_results[key] = np.mean(mean_results[key])

        print("Mean results: ", mean_results)


if __name__ == '__main__':
    main()