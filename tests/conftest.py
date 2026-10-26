# tests/fixtures/ holds sample repos for the agent to work on. Their tests are
# not ours (the toy repo's fail on purpose), so pytest must not collect them.
collect_ignore = ["fixtures"]
