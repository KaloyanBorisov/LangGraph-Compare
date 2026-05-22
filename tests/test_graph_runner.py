from langgraph_compare.graph_runner import run_multiple_iterations

def test_run_graph_iterations(mock_state_graph, capsys):
    """
    Test the `run_graph_iterations` function with a mock `StateGraph`, capturing printed output.

    This test ensures the function:
    - Runs the specified number of repetitions.
    - Configures the `StateGraph` with correct thread IDs and recursion limits.
    - Produces the expected printed output.
    - Calls the `stream` method with correct arguments.

    :param mock_state_graph: A mock StateGraph object provided by the fixture.
    :type mock_state_graph: MagicMock
    :param capsys: A pytest fixture to capture standard output.
    :type capsys: pytest.CaptureFixture
    :raises AssertionError: If the output or behavior does not match expectations.
    """
    # Test parameters
    starting_thread_id = 1
    num_repetitions = 2
    user_input_template = {"input_key": "input_value"}
    recursion_limit = 50

    # Run the function with mock graph and parameters
    run_multiple_iterations(graph=mock_state_graph, starting_thread_id=starting_thread_id,
                            num_repetitions=num_repetitions, user_input_template=user_input_template,
                            recursion_limit=recursion_limit)

    # Capture printed output
    captured = capsys.readouterr()

    # Check the output contains expected print statements
    assert f"Iteration: 1, Thread_ID {starting_thread_id}" in captured.out
    assert f"Iteration: 2, Thread_ID {starting_thread_id + 1}" in captured.out
    assert "output_1" in captured.out
    assert "output_2" in captured.out
    assert "---" in captured.out

    # Verify that `stream` was called with the correct configurations
    expected_config1 = {
        "configurable": {"thread_id": str(starting_thread_id)},
        "recursion_limit": recursion_limit
    }
    expected_config2 = {
        "configurable": {"thread_id": str(starting_thread_id + 1)},
        "recursion_limit": recursion_limit
    }
    mock_state_graph.stream.assert_any_call(user_input_template, expected_config1, stream_mode="values")
    mock_state_graph.stream.assert_any_call(user_input_template, expected_config2, stream_mode="values")
    assert mock_state_graph.stream.call_count == num_repetitions, f"Expected {num_repetitions} calls to stream(), but got {mock_state_graph.stream.call_count}"