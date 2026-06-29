from engine.game_stats import TableStatsTracker


def test_table_stats_counts_public_preflop_rates_once_per_player_hand():
    tracker = TableStatsTracker()
    tracker.observe_event({
        "type": "hand_start",
        "tournament_id": 1,
        "table_id": 1,
        "hand_id": 1,
        "players": [{"name": "opener"}, {"name": "caller"}, {"name": "folder"}],
    })
    tracker.observe_event({
        "type": "action",
        "tournament_id": 1,
        "table_id": 1,
        "hand_id": 1,
        "player": "opener",
        "action": "raise",
        "amount": 100,
        "street": "preflop",
        "call_amount": 20,
    })
    tracker.observe_event({
        "type": "action",
        "tournament_id": 1,
        "table_id": 1,
        "hand_id": 1,
        "player": "caller",
        "action": "call",
        "amount": 100,
        "street": "preflop",
        "call_amount": 100,
    })
    tracker.observe_event({
        "type": "action",
        "tournament_id": 1,
        "table_id": 1,
        "hand_id": 1,
        "player": "folder",
        "action": "fold",
        "amount": 0,
        "street": "preflop",
        "call_amount": 100,
    })

    stats = tracker.snapshot()

    assert stats["hands_observed"] == 1
    assert stats["player_hands_observed"] == 3
    assert stats["action_total"] == 3
    assert stats["vpip"] == 2 / 3
    assert stats["pfr"] == 1 / 3
    assert stats["preflop_call_rate"] == 1 / 3

    opener_stats = tracker.player_snapshot("opener")
    caller_stats = tracker.player_snapshot("caller")
    folder_stats = tracker.player_snapshot("folder")

    assert opener_stats["hands_observed"] == 1
    assert opener_stats["vpip"] == 1.0
    assert opener_stats["pfr"] == 1.0
    assert caller_stats["vpip"] == 1.0
    assert caller_stats["pfr"] == 0.0
    assert caller_stats["preflop_call_rate"] == 1.0
    assert folder_stats["vpip"] == 0.0
    assert folder_stats["pfr"] == 0.0


def test_table_stats_counts_three_bet_and_postflop_aggression():
    tracker = TableStatsTracker()
    tracker.observe_event({
        "type": "hand_start",
        "tournament_id": 1,
        "table_id": 1,
        "hand_id": 1,
        "players": [{"name": "opener"}, {"name": "threebettor"}],
    })
    tracker.observe_event({
        "type": "action",
        "tournament_id": 1,
        "table_id": 1,
        "hand_id": 1,
        "player": "opener",
        "action": "raise",
        "amount": 100,
        "street": "preflop",
        "call_amount": 20,
    })
    tracker.observe_event({
        "type": "action",
        "tournament_id": 1,
        "table_id": 1,
        "hand_id": 1,
        "player": "threebettor",
        "action": "raise",
        "amount": 300,
        "street": "preflop",
        "call_amount": 100,
    })
    tracker.observe_event({
        "type": "action",
        "tournament_id": 1,
        "table_id": 1,
        "hand_id": 1,
        "player": "opener",
        "action": "call",
        "amount": 200,
        "street": "preflop",
        "call_amount": 200,
    })
    tracker.observe_event({
        "type": "action",
        "tournament_id": 1,
        "table_id": 1,
        "hand_id": 1,
        "player": "opener",
        "action": "raise",
        "amount": 160,
        "street": "flop",
        "call_amount": 0,
    })
    tracker.observe_event({
        "type": "action",
        "tournament_id": 1,
        "table_id": 1,
        "hand_id": 1,
        "player": "threebettor",
        "action": "call",
        "amount": 160,
        "street": "flop",
        "call_amount": 160,
    })

    stats = tracker.snapshot()
    opener_stats = tracker.player_snapshot("opener")
    threebettor_stats = tracker.player_snapshot("threebettor")

    assert stats["three_bet_rate"] == 1.0
    assert stats["postflop_aggression_factor"] == 1.0
    assert opener_stats["three_bet_rate"] == 0.0
    assert threebettor_stats["three_bet_rate"] == 1.0
