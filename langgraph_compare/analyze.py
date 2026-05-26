import pandas as pd
import pm4py
import numpy as np
from collections import defaultdict

pd.set_option('display.max_columns', None)


#1
def get_starts(event_log: pd.DataFrame) -> dict[str, int]:
    """
    Get the start activities of the event log.

    :param event_log: Event log data.
    :type event_log: pd.DataFrame
    :return: Start activities and their counts.
    :rtype: dict

    **Example:**

    >>> csv_output = "files/examples.csv"
    >>> event_log = load_event_log(csv_output)
    >>> print(get_starts(event_log))
    Event log loaded and formated from file: files/examples.csv
    {'__start__': 3}
    """
    start_activities = pm4py.get_start_activities(event_log)
    return start_activities


def print_starts(event_log: pd.DataFrame) -> None:
    """
    Print the start activities of the event log.

    :param event_log: Event log data.
    :type event_log: pd.DataFrame

    **Example:**

    >>> csv_output = "files/examples.csv"
    >>> event_log = load_event_log(csv_output)
    >>> print_starts(event_log)
    Event log loaded and formated from file: files/examples.csv
    Start activities: {'__start__': 3}
    """
    start_activities = get_starts(event_log)
    print("Start activities:", start_activities)

#2
def get_ends(event_log: pd.DataFrame) -> dict[str, int]:
    """
    Get the end activities of the event log.

    :param event_log: Event log data.
    :type event_log: pd.DataFrame
    :return: End activities and their counts.
    :rtype: dict

    **Example:**

    >>> csv_output = "files/examples.csv"
    >>> event_log = load_event_log(csv_output)
    >>> print(get_ends(event_log))
    Event log loaded and formated from file: files/examples.csv
    {'test_supervisor': 3}
    """
    end_activities = pm4py.get_end_activities(event_log)
    return end_activities


def print_ends(event_log: pd.DataFrame) -> None:
    """
    Print the end activities of the event log.

    :param event_log: Event log data.
    :type event_log: pd.DataFrame

    **Example:**

    >>> csv_output = "files/examples.csv"
    >>> event_log = load_event_log(csv_output)
    >>> print_ends(event_log)
    Event log loaded and formated from file: files/examples.csv
    End activities: {'test_supervisor': 3}
    """
    end_activities = get_ends(event_log)
    print("End activities:", end_activities)

#3
def get_act_counts(event_log: pd.DataFrame) -> dict[str, int]:
    """
    Get the counts of every activity in the event log.

    :param event_log: Event log data.
    :type event_log: pd.DataFrame
    :return: Activity counts.
    :rtype: dict

    **Example:**

    >>> csv_output = "files/examples.csv"
    >>> event_log = load_event_log(csv_output)
    >>> print(get_act_counts(event_log))
    Event log loaded and formated from file: files/examples.csv
    {'__start__': 27, 'ag_supervisor': 27, 'test_supervisor': 27, 'rg_supervisor': 22, 'DocWriter': 8, 'Search': 5, 'WebScraper': 5, 'ChartGenerator': 3, 'NoteTaker': 3}
    """
    activities = pm4py.get_event_attribute_values(event_log, 'concept:name')
    return activities


def print_act_counts(event_log: pd.DataFrame) -> None:
    """
    Print the counts of every activity in the event log.

    :param event_log: Event log data.
    :type event_log: pd.DataFrame

    **Example:**

    >>> csv_output = "files/examples.csv"
    >>> event_log = load_event_log(csv_output)
    >>> print_act_counts(event_log)
    Event log loaded and formated from file: files/examples.csv
    Count of each activity: {'__start__': 27, 'ag_supervisor': 27, 'test_supervisor': 27, 'rg_supervisor': 22, 'DocWriter': 8, 'Search': 5, 'WebScraper': 5, 'ChartGenerator': 3, 'NoteTaker': 3}
    """
    activities = get_act_counts(event_log)
    print("Count of each activity:", activities)

#4
def get_sequences(event_log: pd.DataFrame) -> dict[int, list[str]]:
    """
    Return activity sequences for every case ID.

    :param event_log: Event log data.
    :type event_log: pd.DataFrame
    :return: Mapping of case IDs to their activity sequences.
    :rtype: dict

    **Example:**

    >>> csv_output = "files/examples.csv"
    >>> event_log = load_event_log(csv_output)
    >>> print(get_sequences(event_log))
    Event log loaded and formated from file: files/examples.csv
    {18: ['__start__', 'ag_supervisor', 'test_supervisor'], 19: ['__start__', 'test_supervisor', '__start__', 'rg_supervisor', 'Search', 'rg_supervisor', 'WebScraper', 'rg_supervisor', 'test_supervisor', '__start__', 'rg_supervisor', 'WebScraper', 'rg_supervisor', 'test_supervisor', '__start__', 'ag_supervisor', 'test_supervisor', '__start__', 'ag_supervisor', 'test_supervisor', '__start__', 'ag_supervisor', 'ChartGenerator', 'ag_supervisor', 'test_supervisor', '__start__', 'rg_supervisor', 'test_supervisor', '__start__', 'ag_supervisor', 'DocWriter', 'ag_supervisor', 'test_supervisor', '__start__', 'rg_supervisor', 'Search', 'rg_supervisor', 'WebScraper', 'rg_supervisor', 'test_supervisor', '__start__', 'rg_supervisor', 'Search', 'rg_supervisor', 'WebScraper', 'rg_supervisor', 'test_supervisor', '__start__', 'ag_supervisor', 'DocWriter', 'ag_supervisor', 'DocWriter', 'ag_supervisor', 'DocWriter', 'ag_supervisor', 'test_supervisor', '__start__', 'rg_supervisor', 'test_supervisor', '__start__', 'ag_supervisor', 'test_supervisor', '__start__', 'ag_supervisor', 'test_supervisor', '__start__', 'ag_supervisor', 'test_supervisor', '__start__', 'ag_supervisor', 'test_supervisor', '__start__', 'rg_supervisor', 'test_supervisor', '__start__', 'rg_supervisor', 'test_supervisor'], 20: ['__start__', 'test_supervisor', '__start__', 'rg_supervisor', 'Search', 'rg_supervisor', 'WebScraper', 'rg_supervisor', 'test_supervisor', '__start__', 'ag_supervisor', 'ChartGenerator', 'ag_supervisor', 'ChartGenerator', 'ag_supervisor', 'test_supervisor', '__start__', 'ag_supervisor', 'DocWriter', 'ag_supervisor', 'test_supervisor', '__start__', 'rg_supervisor', 'test_supervisor', '__start__', 'ag_supervisor', 'DocWriter', 'ag_supervisor', 'NoteTaker', 'ag_supervisor', 'DocWriter', 'ag_supervisor', 'NoteTaker', 'ag_supervisor', 'DocWriter', 'ag_supervisor', 'NoteTaker', 'ag_supervisor', 'test_supervisor', '__start__', 'rg_supervisor', 'Search', 'rg_supervisor', 'test_supervisor', '__start__', 'rg_supervisor', 'test_supervisor']}
    """
    # Tworzymy słownik, który będzie przechowywał sekwencje aktywności dla każdego case_id
    sequences_by_case = defaultdict(list)

    # Budujemy sekwencje dla każdego case_id
    for _, row in event_log.iterrows():
        case_id = int(row['case_id'])
        activity = row['activity']
        sequences_by_case[case_id].append(activity)

    return dict(sorted(sequences_by_case.items()))


def print_sequences(event_log: pd.DataFrame) -> None:
    """
    Print sequences for every case ID in the event log, sorted numerically by case ID.

    :param event_log: Event log data.
    :type event_log: pd.DataFrame

    **Example:**

    >>> csv_output = "files/examples.csv"
    >>> event_log = load_event_log(csv_output)
    >>> print_sequences(event_log)
    Event log loaded and formated from file: files/examples.csv
    All sequences:
    Case ID 18: ['__start__', 'ag_supervisor', 'test_supervisor']
    Case ID 19: ['__start__', 'test_supervisor', '__start__', 'rg_supervisor', 'Search', 'rg_supervisor', 'WebScraper', 'rg_supervisor', 'test_supervisor', '__start__', 'rg_supervisor', 'WebScraper', 'rg_supervisor', 'test_supervisor', '__start__', 'ag_supervisor', 'test_supervisor', '__start__', 'ag_supervisor', 'test_supervisor', '__start__', 'ag_supervisor', 'ChartGenerator', 'ag_supervisor', 'test_supervisor', '__start__', 'rg_supervisor', 'test_supervisor', '__start__', 'ag_supervisor', 'DocWriter', 'ag_supervisor', 'test_supervisor', '__start__', 'rg_supervisor', 'Search', 'rg_supervisor', 'WebScraper', 'rg_supervisor', 'test_supervisor', '__start__', 'rg_supervisor', 'Search', 'rg_supervisor', 'WebScraper', 'rg_supervisor', 'test_supervisor', '__start__', 'ag_supervisor', 'DocWriter', 'ag_supervisor', 'DocWriter', 'ag_supervisor', 'DocWriter', 'ag_supervisor', 'test_supervisor', '__start__', 'rg_supervisor', 'test_supervisor', '__start__', 'ag_supervisor', 'test_supervisor', '__start__', 'ag_supervisor', 'test_supervisor', '__start__', 'ag_supervisor', 'test_supervisor', '__start__', 'ag_supervisor', 'test_supervisor', '__start__', 'rg_supervisor', 'test_supervisor', '__start__', 'rg_supervisor', 'test_supervisor']
    Case ID 20: ['__start__', 'test_supervisor', '__start__', 'rg_supervisor', 'Search', 'rg_supervisor', 'WebScraper', 'rg_supervisor', 'test_supervisor', '__start__', 'ag_supervisor', 'ChartGenerator', 'ag_supervisor', 'ChartGenerator', 'ag_supervisor', 'test_supervisor', '__start__', 'ag_supervisor', 'DocWriter', 'ag_supervisor', 'test_supervisor', '__start__', 'rg_supervisor', 'test_supervisor', '__start__', 'ag_supervisor', 'DocWriter', 'ag_supervisor', 'NoteTaker', 'ag_supervisor', 'DocWriter', 'ag_supervisor', 'NoteTaker', 'ag_supervisor', 'DocWriter', 'ag_supervisor', 'NoteTaker', 'ag_supervisor', 'test_supervisor', '__start__', 'rg_supervisor', 'Search', 'rg_supervisor', 'test_supervisor', '__start__', 'rg_supervisor', 'test_supervisor']
    """
    # Sekwencja dla każdego case_id
    sequences_by_case = get_sequences(event_log)

    print("All sequences:")

    # Sortowanie po case_id
    for case_id in sorted(sequences_by_case.keys(), key=lambda x: int(x)):
        sequence = sequences_by_case[case_id]
        print(f"Case ID {case_id}: {sequence}")

#5
def get_sequence_probs(event_log: pd.DataFrame) -> list[tuple[int, tuple[str, ...], float]]:
    """
    Return sequences with probabilities for each case ID.
    If sequence already occurred it only lists the ID of the latest occurrence.

    :param event_log: Event log data.
    :type event_log: pd.DataFrame
    :return: List of tuples containing (case ID, sequence, probability).
    :rtype: list

    **Example:**

    >>> csv_output = "files/examples.csv"
    >>> event_log = load_event_log(csv_output)
    >>> print(get_sequence_probs(event_log))
    Event log loaded and formated from file: files/examples.csv
    [(18, ('__start__', 'ag_supervisor', 'test_supervisor'), 0.3333333333333333), (19, ('__start__', 'test_supervisor', '__start__', 'rg_supervisor', 'Search', 'rg_supervisor', 'WebScraper', 'rg_supervisor', 'test_supervisor', '__start__', 'rg_supervisor', 'WebScraper', 'rg_supervisor', 'test_supervisor', '__start__', 'ag_supervisor', 'test_supervisor', '__start__', 'ag_supervisor', 'test_supervisor', '__start__', 'ag_supervisor', 'ChartGenerator', 'ag_supervisor', 'test_supervisor', '__start__', 'rg_supervisor', 'test_supervisor', '__start__', 'ag_supervisor', 'DocWriter', 'ag_supervisor', 'test_supervisor', '__start__', 'rg_supervisor', 'Search', 'rg_supervisor', 'WebScraper', 'rg_supervisor', 'test_supervisor', '__start__', 'rg_supervisor', 'Search', 'rg_supervisor', 'WebScraper', 'rg_supervisor', 'test_supervisor', '__start__', 'ag_supervisor', 'DocWriter', 'ag_supervisor', 'DocWriter', 'ag_supervisor', 'DocWriter', 'ag_supervisor', 'test_supervisor', '__start__', 'rg_supervisor', 'test_supervisor', '__start__', 'ag_supervisor', 'test_supervisor', '__start__', 'ag_supervisor', 'test_supervisor', '__start__', 'ag_supervisor', 'test_supervisor', '__start__', 'ag_supervisor', 'test_supervisor', '__start__', 'rg_supervisor', 'test_supervisor', '__start__', 'rg_supervisor', 'test_supervisor'), 0.3333333333333333), (20, ('__start__', 'test_supervisor', '__start__', 'rg_supervisor', 'Search', 'rg_supervisor', 'WebScraper', 'rg_supervisor', 'test_supervisor', '__start__', 'ag_supervisor', 'ChartGenerator', 'ag_supervisor', 'ChartGenerator', 'ag_supervisor', 'test_supervisor', '__start__', 'ag_supervisor', 'DocWriter', 'ag_supervisor', 'test_supervisor', '__start__', 'rg_supervisor', 'test_supervisor', '__start__', 'ag_supervisor', 'DocWriter', 'ag_supervisor', 'NoteTaker', 'ag_supervisor', 'DocWriter', 'ag_supervisor', 'NoteTaker', 'ag_supervisor', 'DocWriter', 'ag_supervisor', 'NoteTaker', 'ag_supervisor', 'test_supervisor', '__start__', 'rg_supervisor', 'Search', 'rg_supervisor', 'test_supervisor', '__start__', 'rg_supervisor', 'test_supervisor'), 0.3333333333333333)]
    """
    sequences_by_case = get_sequences(event_log)

    # Generujemy probabilistyczny język
    language = pm4py.get_stochastic_language(event_log)

    # Tworzymy odwrotny słownik dla łatwego porównania
    case_by_sequence = {tuple(seq): case_id for case_id, seq in sequences_by_case.items()}

    # Sortujemy sequences_by_case według numerycznych wartości case_id
    sorted_sequences = sorted(case_by_sequence.items(), key=lambda x: int(x[1]))

    # Generowanie listy rezultatów
    result = [(case_id, sequence, language.get(sequence, 0)) for sequence, case_id in sorted_sequences]
    return result


def print_sequence_probs(event_log: pd.DataFrame) -> None:
    """
    Print sequences with probabilities for each case ID.
    If sequence already occurred it only prints the ID of the latest occurrence.

    :param event_log: Event log data.
    :type event_log: pd.DataFrame

    **Example:**

    >>> csv_output = "files/examples.csv"
    >>> event_log = load_event_log(csv_output)
    >>> print_sequence_probs(event_log)
    Event log loaded and formated from file: files/examples.csv
    ID of last sequence occurrence with probability of occurrence:
    Case ID 18: ('__start__', 'ag_supervisor', 'test_supervisor')
    Probability: 0.333
    Case ID 19: ('__start__', 'test_supervisor', '__start__', 'rg_supervisor', 'Search', 'rg_supervisor', 'WebScraper', 'rg_supervisor', 'test_supervisor', '__start__', 'rg_supervisor', 'WebScraper', 'rg_supervisor', 'test_supervisor', '__start__', 'ag_supervisor', 'test_supervisor', '__start__', 'ag_supervisor', 'test_supervisor', '__start__', 'ag_supervisor', 'ChartGenerator', 'ag_supervisor', 'test_supervisor', '__start__', 'rg_supervisor', 'test_supervisor', '__start__', 'ag_supervisor', 'DocWriter', 'ag_supervisor', 'test_supervisor', '__start__', 'rg_supervisor', 'Search', 'rg_supervisor', 'WebScraper', 'rg_supervisor', 'test_supervisor', '__start__', 'rg_supervisor', 'Search', 'rg_supervisor', 'WebScraper', 'rg_supervisor', 'test_supervisor', '__start__', 'ag_supervisor', 'DocWriter', 'ag_supervisor', 'DocWriter', 'ag_supervisor', 'DocWriter', 'ag_supervisor', 'test_supervisor', '__start__', 'rg_supervisor', 'test_supervisor', '__start__', 'ag_supervisor', 'test_supervisor', '__start__', 'ag_supervisor', 'test_supervisor', '__start__', 'ag_supervisor', 'test_supervisor', '__start__', 'ag_supervisor', 'test_supervisor', '__start__', 'rg_supervisor', 'test_supervisor', '__start__', 'rg_supervisor', 'test_supervisor')
    Probability: 0.333
    Case ID 20: ('__start__', 'test_supervisor', '__start__', 'rg_supervisor', 'Search', 'rg_supervisor', 'WebScraper', 'rg_supervisor', 'test_supervisor', '__start__', 'ag_supervisor', 'ChartGenerator', 'ag_supervisor', 'ChartGenerator', 'ag_supervisor', 'test_supervisor', '__start__', 'ag_supervisor', 'DocWriter', 'ag_supervisor', 'test_supervisor', '__start__', 'rg_supervisor', 'test_supervisor', '__start__', 'ag_supervisor', 'DocWriter', 'ag_supervisor', 'NoteTaker', 'ag_supervisor', 'DocWriter', 'ag_supervisor', 'NoteTaker', 'ag_supervisor', 'DocWriter', 'ag_supervisor', 'NoteTaker', 'ag_supervisor', 'test_supervisor', '__start__', 'rg_supervisor', 'Search', 'rg_supervisor', 'test_supervisor', '__start__', 'rg_supervisor', 'test_supervisor')
    Probability: 0.333
    """
    sequences_with_probabilities = get_sequence_probs(event_log)

    print("ID of last sequence occurrence with probability of occurrence:")
    for case_id, sequence, probability in sequences_with_probabilities:
        print(f"Case ID {case_id}: {sequence}")
        print(f"Probability: {round(probability,3)}\n")

#6
def get_min_self_dists(event_log: pd.DataFrame) -> dict[int, dict[str, int]]:
    """
    Calculate the minimum self-distances for each activity in each case.

    :param event_log: Event log data.
    :type event_log: pd.DataFrame
    :return: Dictionary where keys are case IDs and values are dictionaries of activities with their minimum self-distances.
    :rtype: dict

    **Example:**

    >>> csv_output = "files/examples.csv"
    >>> event_log = load_event_log(csv_output)
    >>> print(get_min_self_dists(event_log))
    Event log loaded and formated from file: files/examples.csv
    {18: {}, 19: {'DocWriter': 1, 'Search': 6, 'WebScraper': 4, '__start__': 1, 'ag_supervisor': 1, 'rg_supervisor': 1, 'test_supervisor': 2}, 20: {'ChartGenerator': 1, 'DocWriter': 3, 'NoteTaker': 3, 'Search': 36, '__start__': 1, 'ag_supervisor': 1, 'rg_supervisor': 1, 'test_supervisor': 2}}
    """
    unique_case_ids = event_log['case:concept:name'].unique()
    sorted_case_ids = np.sort(unique_case_ids.astype(int))

    # Słownik do przechowywania minimalnych odległości własnych dla każdego Case ID
    min_self_distances = {}

    # Iterujemy po każdym case_id i obliczamy minimalne odległości własne dla jego aktywności
    for case_id in sorted_case_ids:
        # Filtrowanie event logu dla danego case_id
        filtered_event_log = event_log[event_log['case:concept:name'] == str(case_id)]

        # Kalkulowanie minimalnych odległości własnych dla danego case_id
        msd = pm4py.get_minimum_self_distances(
            filtered_event_log,
            activity_key='concept:name',
            case_id_key='case:concept:name',
            timestamp_key='time:timestamp'
        )
        min_self_distances[int(case_id)] = msd

    return min_self_distances


def print_min_self_dists(event_log: pd.DataFrame) -> None:
    """
    Calculate and print the minimum self-distances for each activity in each case.

    :param event_log: Event log data.
    :type event_log: pd.DataFrame

    **Example:**

    >>> csv_output = "files/examples.csv"
    >>> event_log = load_event_log(csv_output)
    >>> print_min_self_dists(event_log)
    Event log loaded and formated from file: files/examples.csv
    Minimal self-distances for every activity:
    Case ID 18: {}
    Case ID 19: {'DocWriter': 1, 'Search': 6, 'WebScraper': 4, '__start__': 1, 'ag_supervisor': 1, 'rg_supervisor': 1, 'test_supervisor': 2}
    Case ID 20: {'ChartGenerator': 1, 'DocWriter': 3, 'NoteTaker': 3, 'Search': 36, '__start__': 1, 'ag_supervisor': 1, 'rg_supervisor': 1, 'test_supervisor': 2}
    """
    min_self_distances = get_min_self_dists(event_log)

    print("Minimal self-distances for every activity:")
    for case_id, distances in min_self_distances.items():
        print(f"Case ID {case_id}: {distances}")

#7
def get_act_reworks(event_log: pd.DataFrame) -> dict[int, dict[str, int]]:
    """
    Return the rework counts for each activity in each case.

    :param event_log: Event log data.
    :type event_log: pd.DataFrame
    :return: Rework counts for each case ID.
    :rtype: dict

    **Example:**

    >>> csv_output = "files/examples.csv"
    >>> event_log = load_event_log(csv_output)
    >>> print(get_global_act_reworks(event_log))
    Event log loaded and formated from file: files/examples.csv
    {18: {}, 19: {'__start__': 18, 'test_supervisor': 18, 'rg_supervisor': 15, 'Search': 3, 'WebScraper': 4, 'ag_supervisor': 14, 'DocWriter': 4}, 20: {'__start__': 8, 'test_supervisor': 8, 'rg_supervisor': 7, 'Search': 2, 'ag_supervisor': 12, 'ChartGenerator': 2, 'DocWriter': 4, 'NoteTaker': 3}}
    """
    rework_counts_by_case = {}
    unique_case_ids = event_log['case:concept:name'].unique()
    sorted_case_ids = np.sort(unique_case_ids.astype(int))

    for case_id in sorted_case_ids:
        # Filtrowanie logu dla bieżącego case_id
        filtered_event_log = event_log[event_log['case:concept:name'] == str(case_id)]

        # Inicjalizacja licznika powtórzeń
        activity_counts = defaultdict(int)

        # Tworzymy listę aktywności w bieżącej ścieżce
        activities = filtered_event_log['concept:name'].tolist()

        # Zliczanie wystąpień każdej aktywności
        for activity in activities:
            activity_counts[activity] += 1

        # Usunięcie aktywności, które wystąpiły tylko raz (bo nie są "rework")
        rework_counts = {activity: count for activity, count in activity_counts.items() if count > 1}
        rework_counts_by_case[int(case_id)] = rework_counts
    return rework_counts_by_case


def print_act_reworks(event_log: pd.DataFrame) -> None:
    """
    Print the rework counts for each activity in each case.

    :param event_log: Event log data.
    :type event_log: pd.DataFrame

    **Example:**

    >>> csv_output = "files/examples.csv"
    >>> event_log = load_event_log(csv_output)
    >>> print_act_reworks(event_log)
    Event log loaded and formated from file: files/examples.csv
    Count of activity rework:
    Case ID 18: {}
    Case ID 19: {'__start__': 18, 'test_supervisor': 18, 'rg_supervisor': 15, 'Search': 3, 'WebScraper': 4, 'ag_supervisor': 14, 'DocWriter': 4}
    Case ID 20: {'__start__': 8, 'test_supervisor': 8, 'rg_supervisor': 7, 'Search': 2, 'ag_supervisor': 12, 'ChartGenerator': 2, 'DocWriter': 4, 'NoteTaker': 3}
    """
    rework_counts_by_case = get_act_reworks(event_log)

    print("Count of activity rework:")
    for case_id, rework_counts in rework_counts_by_case.items():
        print(f"Case ID {case_id}: {rework_counts}")

#8
def get_global_act_reworks(event_log: pd.DataFrame) -> dict[str, int]:
    """
    Return the global rework counts for each activity by summing reworks from each case.
    A rework is counted when an activity appears more than once within the same case.

    :param event_log: Event log data.
    :type event_log: pd.DataFrame
    :return: Global rework counts for each activity.
    :rtype: dict

    **Example:**

    >>> csv_output = "files/examples.csv"
    >>> event_log = load_event_log(csv_output)
    >>> print(get_global_act_reworks(event_log))
    Event log loaded and formated from file: files/examples.csv
    {'__start__': 24, 'test_supervisor': 24, 'rg_supervisor': 20, 'Search': 3, 
     'WebScraper': 2, 'ag_supervisor': 24, 'DocWriter': 6, 'ChartGenerator': 1, 
     'NoteTaker': 2}
    """
    # Initialize global rework counter
    global_rework_counts = defaultdict(int)

    # Get unique case IDs
    unique_case_ids = event_log['case:concept:name'].unique()

    for case_id in unique_case_ids:
        # Filter log for current case_id
        case_events = event_log[event_log['case:concept:name'] == case_id]

        # Count activities within this case
        case_activity_counts = defaultdict(int)

        # Count occurrences for each activity in this case
        for activity in case_events['concept:name']:
            case_activity_counts[activity] += 1

        # For each activity that appears more than once in this case,
        # add the number of extra occurrences (reworks) to the global count
        for activity, count in case_activity_counts.items():
            if count > 1:
                # Only count the extra occurrences (subtract 1 from total count)
                global_rework_counts[activity] += (count - 1)

    # Convert defaultdict to regular dict for return
    return dict(global_rework_counts)


def print_global_act_reworks(event_log: pd.DataFrame) -> None:
    """
    Return the global rework counts for each activity across all cases.

    :param event_log: Event log data.
    :type event_log: pd.DataFrame

    **Example:**

    >>> csv_output = "files/examples.csv"
    >>> event_log = load_event_log(csv_output)
    >>> print_global_act_reworks(event_log)
    Event log loaded and formated from file: files/examples.csv
    Global rework counts for each activity:
    Activity '__start__': 11
    Activity 'test_supervisor': 11
    Activity 'rg_supervisor': 9
    Activity 'ag_supervisor': 21
    Activity 'NoteTaker': 3
    Activity 'ChartGenerator': 3
    Activity 'DocWriter': 4
    """
    rework_counts_by_case = get_global_act_reworks(event_log)

    print("Global rework counts for each activity:")
    for activity, rework_counts in rework_counts_by_case.items():
        print(f"Activity '{activity}': {rework_counts}")

#9
def get_mean_act_times(event_log: pd.DataFrame) -> dict[str, float]:
    """
    Calculate the mean service time for each activity.

    :param event_log: Event log data.
    :type event_log: pd.DataFrame
    :return: Mean service times for each activity.
    :rtype: dict

    **Example:**

    >>> csv_output = "files/examples.csv"
    >>> event_log = load_event_log(csv_output)
    >>> print(get_mean_act_times(event_log))
    Event log loaded and formated from file: files/examples.csv
    {'ChartGenerator': 0.587241, 'DocWriter': 1.0209089999999998, 'NoteTaker': 0.5753873333333334, 'Search': 0.580575, 'WebScraper': 0.6020846, '__start__': 0.0411957037037037, 'ag_supervisor': 0.007210296296296296, 'rg_supervisor': 1.8212668636363636, 'test_supervisor': 0.04827048148148148}
    """
    mean_serv_time = pm4py.get_service_time(
        event_log, start_timestamp_key='timestamp', timestamp_key='end_timestamp', aggregation_measure='mean'
    )
    return mean_serv_time


def print_mean_act_times(event_log: pd.DataFrame) -> None:
    """
    Print the mean service time for each activity.

    :param event_log: Event log data.
    :type event_log: pd.DataFrame

    **Example:**

    >>> csv_output = "files/examples.csv"
    >>> event_log = load_event_log(csv_output)
    >>> print_mean_act_times(event_log)
    Event log loaded and formated from file: files/examples.csv
    Mean duration of every activity:
    Activity 'ChartGenerator': 0.587 s
    Activity 'DocWriter': 1.021 s
    Activity 'NoteTaker': 0.575 s
    Activity 'Search': 0.581 s
    Activity 'WebScraper': 0.602 s
    Activity '__start__': 0.0412 s
    Activity 'ag_supervisor': 0.007 s
    Activity 'rg_supervisor': 1.821 s
    Activity 'test_supervisor': 0.048 s
    """
    mean_serv_time = get_mean_act_times(event_log)
    print("Mean duration of every activity:")
    for activity, time in mean_serv_time.items():
        print(f"Activity '{activity}': {round(time,5)} s")

#10
def get_durations(event_log: pd.DataFrame) -> dict[str, float]:
    """
    Calculate the duration of each case in seconds.

    :param event_log: Event log data.
    :type event_log: pd.DataFrame
    :return: Case durations.
    :rtype: dict

    **Example:**

    >>> csv_output = "files/examples.csv"
    >>> event_log = load_event_log(csv_output)
    >>> print(get_durations(event_log))
    Event log loaded and formated from file: files/examples.csv
    {'18': 4.580137, '19': 120.730501, '20': 74.653202}
    """
    # Pobranie unikalnych identyfikatorów przypadków (case_id)
    unique_case_ids = event_log['case:concept:name'].unique()
    sorted_array = np.sort(unique_case_ids.astype(int)).astype(str)

    # Iterowanie po każdym case_id i uzyskanie jego czasu trwania
    case_durations = {}
    for case_id in sorted_array:
        duration = pm4py.get_case_duration(event_log, case_id)
        case_durations[case_id] = duration
    return case_durations


def print_durations(event_log: pd.DataFrame) -> None:
    """
    Print the duration of each case in seconds.

    :param event_log: Event log data.
    :type event_log: pd.DataFrame

    **Example:**

    >>> csv_output = "files/examples.csv"
    >>> event_log = load_event_log(csv_output)
    >>> print_durations(event_log)
    Event log loaded and formated from file: files/examples.csv
    Duration of the case:
    Case ID 18: 4.580 s
    Case ID 19: 120.731 s
    Case ID 20: 74.653 s
    """
    case_durations = get_durations(event_log)
    print("Duration of the case:")

    for case_id, duration in case_durations.items():
        print(f"Case ID {case_id}: {round(duration,5)} s")

#11
def get_avg_duration(event_log: pd.DataFrame) -> float:
    """
    Calculate average duration of the case in seconds.

    :param event_log: Event log data.
    :type event_log: pd.DataFrame

    **Example:**

    >>> csv_output = "files/examples.csv"
    >>> event_log = load_event_log(csv_output)
    >>> print(get_avg_duration(event_log))
    Event log loaded and formated from file: files/examples.csv
    91.56
    """
    duration = pm4py.get_all_case_durations(event_log)
    if not duration:
        return 0.0
    return sum(duration) / len(duration)


def print_avg_duration(event_log: pd.DataFrame) -> None:
    """
    Print the average duration of the case in seconds.

    :param event_log: Event log data.
    :type event_log: pd.DataFrame

    **Example:**

    >>> csv_output = "files/examples.csv"
    >>> event_log = load_event_log(csv_output)
    >>> print_avg_duration(event_log)
    Event log loaded and formated from file: files/examples.csv
    Average case duration: 91.56 s
    """
    duration = get_avg_duration(event_log)
    print(f"Average case duration: {round(duration,5)} s.")

#12
def get_self_dist_witnesses(event_log: pd.DataFrame) -> dict[int, dict[str, list[list[str]]]]:
    """
    Compute the minimum self-distance witnesses for each activity in each case of the event log,
    considering both activity name and resource.

    :param event_log: Event log data containing events with case IDs, activity names, and resources
    :type event_log: pd.DataFrame
    :return: A dictionary where each key is a case ID, and each value is another dictionary mapping
             activities to lists of witness sequences.
    :rtype: dict

    **Example:**

    >>> csv_output = "files/examples.csv"
    >>> event_log = load_event_log(csv_output)
    >>> print(get_all_self_distance_witnesses(event_log))
    Event log loaded and formated from file: files/examples.csv
    {18: {}, 19: {'__start__': [['test_supervisor']], 'test_supervisor': [['__start__', 'rg_supervisor'], ['__start__', 'ag_supervisor']], 'rg_supervisor': [['Search'], ['WebScraper']], 'Search': [['rg_supervisor', 'WebScraper', 'rg_supervisor', 'test_supervisor', '__start__', 'rg_supervisor']], 'WebScraper': [['rg_supervisor', 'test_supervisor', '__start__', 'rg_supervisor']], 'ag_supervisor': [['DocWriter'], ['ChartGenerator']], 'DocWriter': [['ag_supervisor']]}, 20: {'__start__': [['test_supervisor']], 'test_supervisor': [['__start__', 'rg_supervisor']], 'rg_supervisor': [['Search'], ['WebScraper']], 'Search': [['rg_supervisor', 'WebScraper', 'rg_supervisor', 'test_supervisor', '__start__', 'ag_supervisor', 'ChartGenerator', 'ag_supervisor', 'ChartGenerator', 'ag_supervisor', 'test_supervisor', '__start__', 'ag_supervisor', 'DocWriter', 'ag_supervisor', 'test_supervisor', '__start__', 'rg_supervisor', 'test_supervisor', '__start__', 'ag_supervisor', 'DocWriter', 'ag_supervisor', 'NoteTaker', 'ag_supervisor', 'DocWriter', 'ag_supervisor', 'NoteTaker', 'ag_supervisor', 'DocWriter', 'ag_supervisor', 'NoteTaker', 'ag_supervisor', 'test_supervisor', '__start__', 'rg_supervisor']], 'ag_supervisor': [['DocWriter'], ['ChartGenerator'], ['NoteTaker']], 'ChartGenerator': [['ag_supervisor']], 'DocWriter': [['ag_supervisor', 'NoteTaker', 'ag_supervisor']], 'NoteTaker': [['ag_supervisor', 'DocWriter', 'ag_supervisor']]}}
    """
    # Konwersja case_id do int'a
    event_log['case_id'] = event_log['case_id'].astype(int)
    unique_case_ids = event_log['case_id'].unique()

    # Sortowanie case_id
    sorted_case_ids = sorted(unique_case_ids)

    all_msd_witnesses = {}

    for case_id in sorted_case_ids:
        case_id = int(case_id)
        # Filtrowanie event log'u dla aktualnego case_id
        filtered_event_log = event_log[event_log['case_id'] == case_id].copy()

        if filtered_event_log.empty:
            continue

        # Reset index'u by zapewnić, że index'y zaczynają się od 0
        filtered_event_log.reset_index(drop=True, inplace=True)

        # Znajdź unikalne kombinacje aktywność-zasób
        unique_pairs = []
        for _, row in filtered_event_log.iterrows():
            pair = (row['concept:name'], row['org:resource'])
            if pair not in unique_pairs:
                unique_pairs.append(pair)

        corrected_witnesses = {}

        # Dla każdej unikalnej pary aktywność-zasób
        for activity, resource in unique_pairs:
            # Znajdź indeksy dla tej konkretnej kombinacji aktywność-zasób
            activity_mask = (filtered_event_log['concept:name'] == activity) & (
                        filtered_event_log['org:resource'] == resource)
            indices = filtered_event_log[activity_mask].index.tolist()

            # Pomiń jeśli nie ma przynajmniej dwóch wystąpień
            if len(indices) < 2:
                continue

            # Wylicz przerwy między kolejnymi wystąpieniami
            gaps = []
            consecutive_indices = []

            for i in range(len(indices) - 1):
                gap = indices[i + 1] - indices[i] - 1

                # Sprawdź czy między wystąpieniami nie ma tej samej aktywności z innym zasobem
                events_between = filtered_event_log.iloc[indices[i] + 1:indices[i + 1]]
                if not any((events_between['concept:name'] == activity) & (events_between['org:resource'] != resource)):
                    gaps.append(gap)
                    consecutive_indices.append((indices[i], indices[i + 1]))

            # Jeśli nie ma żadnych właściwych przerw, pomiń tę aktywność
            if not gaps:
                continue

            min_distance = min(gaps)

            # Zidentyfikuj świadków dla minimalnych odległości własnych
            witness_sequences = []
            for start_idx, end_idx in consecutive_indices:
                gap_size = end_idx - start_idx - 1
                if gap_size == min_distance:
                    # Wydobycie eventów pomiędzy
                    gap_events = filtered_event_log.iloc[start_idx + 1:end_idx]['concept:name'].tolist()
                    # Wyłącz aktywność z listy
                    gap_events = [event for event in gap_events if event != activity]
                    # Dodaj tylko nie pustę przerwy
                    if gap_events:
                        witness_sequences.append(gap_events)

            # De duplikacja sekwencji z zachowaniem ich kolejności
            unique_sequences = list(map(list, {tuple(seq) for seq in witness_sequences}))
            if unique_sequences:  # Dodaj tylko jeśli są świadkowie
                corrected_witnesses[activity] = unique_sequences

        all_msd_witnesses[case_id] = corrected_witnesses

    return all_msd_witnesses


def print_self_dist_witnesses(event_log: pd.DataFrame) -> None:
    """
    Print the minimum self-distance witnesses for each activity in each case.

    :param event_log: Event log data.
    :type event_log: pd.DataFrame

    **Example:**

    >>> csv_output = "files/examples.csv"
    >>> event_log = load_event_log(csv_output)
    >>> print_self_dist_witnesses(event_log)
    Event log loaded and formated from file: files/examples.csv
    Witnesses of minimum self-distances:
    Case ID 18: {}
    Case ID 19: {'__start__': [['test_supervisor']], 'test_supervisor': [['__start__', 'rg_supervisor'], ['__start__', 'ag_supervisor']], 'rg_supervisor': [['Search'], ['WebScraper']], 'Search': [['rg_supervisor', 'WebScraper', 'rg_supervisor', 'test_supervisor', '__start__', 'rg_supervisor']], 'WebScraper': [['rg_supervisor', 'test_supervisor', '__start__', 'rg_supervisor']], 'ag_supervisor': [['DocWriter'], ['ChartGenerator']], 'DocWriter': [['ag_supervisor']]}
    Case ID 20: {'__start__': [['test_supervisor']], 'test_supervisor': [['__start__', 'rg_supervisor']], 'rg_supervisor': [['Search'], ['WebScraper']], 'Search': [['rg_supervisor', 'WebScraper', 'rg_supervisor', 'test_supervisor', '__start__', 'ag_supervisor', 'ChartGenerator', 'ag_supervisor', 'ChartGenerator', 'ag_supervisor', 'test_supervisor', '__start__', 'ag_supervisor', 'DocWriter', 'ag_supervisor', 'test_supervisor', '__start__', 'rg_supervisor', 'test_supervisor', '__start__', 'ag_supervisor', 'DocWriter', 'ag_supervisor', 'NoteTaker', 'ag_supervisor', 'DocWriter', 'ag_supervisor', 'NoteTaker', 'ag_supervisor', 'DocWriter', 'ag_supervisor', 'NoteTaker', 'ag_supervisor', 'test_supervisor', '__start__', 'rg_supervisor']], 'ag_supervisor': [['DocWriter'], ['ChartGenerator'], ['NoteTaker']], 'ChartGenerator': [['ag_supervisor']], 'DocWriter': [['ag_supervisor', 'NoteTaker', 'ag_supervisor']], 'NoteTaker': [['ag_supervisor', 'DocWriter', 'ag_supervisor']]}
    """
    all_msd_witnesses = get_self_dist_witnesses(event_log)

    print("Witnesses of minimum self-distances:")
    for case_id, witnesses in all_msd_witnesses.items():
        print(f"Case ID {case_id}: {witnesses}")


def print_analysis(event_log: pd.DataFrame) -> None:
    """
    Run multiple analyses on the event log and print the results.

    :param event_log: The event log data to analyze.
    :type event_log: pd.DataFrame

    **Example**:

    .. code-block:: python

        csv_output = "files/examples.csv"
        event_log = load_event_log(csv_output)
        print_all_self_distance_witnesses(event_log)
        # Event log loaded and formated from file: files/examples.csv
        #
        # #########################START#########################
        #
        # Start activities: {'__start__': 3}
        #
        # End activities: {'test_supervisor': 3}
        #
        # Count of each activity: {'ag_supervisor': 24, '__start__': 14, 'test_supervisor': 14, 'rg_supervisor': 12, 'DocWriter': 7, 'NoteTaker': 6, 'ChartGenerator': 6, 'Search': 3, 'WebScraper': 3}
        #
        # All sequences:
        # Case ID 1: ['__start__', 'test_supervisor', '__start__', 'rg_supervisor', 'Search', 'rg_supervisor', 'WebScraper', 'rg_supervisor', 'test_supervisor', '__start__', 'ag_supervisor', 'DocWriter', 'ag_supervisor', 'NoteTaker', 'ag_supervisor', 'ChartGenerator', 'ag_supervisor', 'test_supervisor', '__start__', 'rg_supervisor', 'test_supervisor']
        # Case ID 2: ['__start__', 'test_supervisor', '__start__', 'rg_supervisor', 'Search', 'rg_supervisor', 'WebScraper', 'rg_supervisor', 'test_supervisor', '__start__', 'ag_supervisor', 'NoteTaker', 'ag_supervisor', 'ChartGenerator', 'ag_supervisor', 'test_supervisor', '__start__', 'ag_supervisor', 'DocWriter', 'ag_supervisor', 'DocWriter', 'ag_supervisor', 'DocWriter', 'ag_supervisor', 'DocWriter', 'ag_supervisor', 'NoteTaker', 'ag_supervisor', 'NoteTaker', 'ag_supervisor', 'ChartGenerator', 'ag_supervisor', 'DocWriter', 'ag_supervisor', 'NoteTaker', 'ag_supervisor', 'ChartGenerator', 'ag_supervisor', 'test_supervisor', '__start__', 'rg_supervisor', 'test_supervisor', '__start__', 'ag_supervisor', 'test_supervisor']
        # Case ID 3: ['__start__', 'test_supervisor', '__start__', 'rg_supervisor', 'Search', 'rg_supervisor', 'WebScraper', 'rg_supervisor', 'test_supervisor', '__start__', 'rg_supervisor', 'test_supervisor', '__start__', 'ag_supervisor', 'NoteTaker', 'ag_supervisor', 'ChartGenerator', 'ag_supervisor', 'ChartGenerator', 'ag_supervisor', 'DocWriter', 'ag_supervisor', 'test_supervisor']
        #
        # ID of last sequence occurrence with probability of occurrence:
        # Case ID 1: ('__start__', 'test_supervisor', '__start__', 'rg_supervisor', 'Search', 'rg_supervisor', 'WebScraper', 'rg_supervisor', 'test_supervisor', '__start__', 'ag_supervisor', 'DocWriter', 'ag_supervisor', 'NoteTaker', 'ag_supervisor', 'ChartGenerator', 'ag_supervisor', 'test_supervisor', '__start__', 'rg_supervisor', 'test_supervisor')
        # Probability: 0.333
        #
        # Case ID 2: ('__start__', 'test_supervisor', '__start__', 'rg_supervisor', 'Search', 'rg_supervisor', 'WebScraper', 'rg_supervisor', 'test_supervisor', '__start__', 'ag_supervisor', 'NoteTaker', 'ag_supervisor', 'ChartGenerator', 'ag_supervisor', 'test_supervisor', '__start__', 'ag_supervisor', 'DocWriter', 'ag_supervisor', 'DocWriter', 'ag_supervisor', 'DocWriter', 'ag_supervisor', 'DocWriter', 'ag_supervisor', 'NoteTaker', 'ag_supervisor', 'NoteTaker', 'ag_supervisor', 'ChartGenerator', 'ag_supervisor', 'DocWriter', 'ag_supervisor', 'NoteTaker', 'ag_supervisor', 'ChartGenerator', 'ag_supervisor', 'test_supervisor', '__start__', 'rg_supervisor', 'test_supervisor', '__start__', 'ag_supervisor', 'test_supervisor')
        # Probability: 0.333
        #
        # Case ID 3: ('__start__', 'test_supervisor', '__start__', 'rg_supervisor', 'Search', 'rg_supervisor', 'WebScraper', 'rg_supervisor', 'test_supervisor', '__start__', 'rg_supervisor', 'test_supervisor', '__start__', 'ag_supervisor', 'NoteTaker', 'ag_supervisor', 'ChartGenerator', 'ag_supervisor', 'ChartGenerator', 'ag_supervisor', 'DocWriter', 'ag_supervisor', 'test_supervisor')
        # Probability: 0.333
        #
        # Minimal self-distances for every activity:
        # Case ID 1: {'__start__': 1, 'ag_supervisor': 1, 'rg_supervisor': 1, 'test_supervisor': 2}
        # Case ID 2: {'ChartGenerator': 5, 'DocWriter': 1, 'NoteTaker': 1, '__start__': 1, 'ag_supervisor': 1, 'rg_supervisor': 1, 'test_supervisor': 2}
        # Case ID 3: {'ChartGenerator': 1, '__start__': 1, 'ag_supervisor': 1, 'rg_supervisor': 1, 'test_supervisor': 2}
        #
        # Witnesses of minimum self-distances:
        # Case ID 1: {'__start__': [['test_supervisor']], 'test_supervisor': [['__start__', 'rg_supervisor']], 'rg_supervisor': [['WebScraper'], ['Search']], 'ag_supervisor': [['DocWriter'], ['NoteTaker'], ['ChartGenerator']]}
        # Case ID 2: {'__start__': [['test_supervisor']], 'test_supervisor': [['__start__', 'rg_supervisor'], ['__start__', 'ag_supervisor']], 'rg_supervisor': [['WebScraper'], ['Search']], 'ag_supervisor': [['DocWriter'], ['NoteTaker'], ['ChartGenerator']], 'NoteTaker': [['ag_supervisor']], 'ChartGenerator': [['ag_supervisor', 'DocWriter', 'ag_supervisor', 'NoteTaker', 'ag_supervisor']], 'DocWriter': [['ag_supervisor']]}
        # Case ID 3: {'__start__': [['test_supervisor']], 'test_supervisor': [['__start__', 'rg_supervisor']], 'rg_supervisor': [['WebScraper'], ['Search']], 'ag_supervisor': [['DocWriter'], ['NoteTaker'], ['ChartGenerator']], 'ChartGenerator': [['ag_supervisor']]}
        #
        # Count of activity rework:
        # Case ID 1: {'__start__': 4, 'test_supervisor': 4, 'rg_supervisor': 4, 'ag_supervisor': 4}
        # Case ID 2: {'__start__': 6, 'test_supervisor': 6, 'rg_supervisor': 4, 'ag_supervisor': 15, 'NoteTaker': 4, 'ChartGenerator': 3, 'DocWriter': 5}
        # Case ID 3: {'__start__': 4, 'test_supervisor': 4, 'rg_supervisor': 4, 'ag_supervisor': 5, 'ChartGenerator': 2}
        #
        # Mean duration of every activity:
        # Activity 'ChartGenerator': 0.541 s
        # Activity 'DocWriter': 0.523 s
        # Activity 'NoteTaker': 0.606 s
        # Activity 'Search': 1.587 s
        # Activity 'WebScraper': 0.525 s
        # Activity '__start__': 0.288 s
        # Activity 'ag_supervisor': 0.005 s
        # Activity 'rg_supervisor': 0.759 s
        # Activity 'test_supervisor': 0.003 s
        #
        # Duration of the case:
        # Case ID 1: 43.88 s
        # Case ID 2: 62.252 s
        # Case ID 3: 29.688 s
        #
        # #########################END#########################
    """

    print("\n"+"#"*25+"START"+"#"*25+"\n")

    print_starts(event_log)
    print()

    print_ends(event_log)
    print()

    print_act_counts(event_log)
    print()

    print_sequences(event_log)
    print()

    print_sequence_probs(event_log)

    print_min_self_dists(event_log)
    print()

    print_self_dist_witnesses(event_log)
    print()

    print_act_reworks(event_log)
    print()

    print_mean_act_times(event_log)
    print()

    print_durations(event_log)

    print("\n"+"#"*25+"END"+"#"*25)
