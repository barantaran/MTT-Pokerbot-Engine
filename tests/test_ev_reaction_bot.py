from players.ev_reaction_bot import EVInitiativeBot, EVReactionBot


def test_ev_reaction_bot_folds_negative_call_ev_with_visible_equity():
    bot = EVReactionBot()

    action = bot.get_action({"hero_equity": 0.30, "pot_size": 1000, "call_amount": 500})

    assert action == ("fold", 0)


def test_ev_reaction_bot_calls_positive_call_ev_with_visible_equity():
    bot = EVReactionBot()

    action = bot.get_action({"hero_equity": 0.62, "pot_size": 1000, "call_amount": 500})

    assert action == ("call", 0)


def test_ev_reaction_bot_checks_free_action():
    bot = EVReactionBot()

    action = bot.get_action({"hero_equity": 0.10, "pot_size": 1000, "call_amount": 0})

    assert action == ("call", 0)


def test_ev_initiative_bot_folds_negative_call_ev():
    bot = EVInitiativeBot()

    action = bot.get_action({"hero_equity": 0.30, "pot_size": 1000, "call_amount": 500})

    assert action == ("fold", 0)


def test_ev_initiative_bot_calls_positive_ev_without_enough_raise_edge():
    bot = EVInitiativeBot()

    action = bot.get_action({
        "hero_equity": 0.52,
        "pot_size": 100,
        "call_amount": 100,
        "stack_size": 5000,
        "min_raise": 100,
        "blinds": {"big": 50},
        "active_players": 2,
        "board_cards": [],
    })

    assert action == ("call", 0)


def test_ev_initiative_bot_raises_size_from_positive_ev_amount():
    bot = EVInitiativeBot()

    action = bot.get_action({
        "hero_equity": 0.70,
        "pot_size": 1000,
        "call_amount": 100,
        "stack_size": 5000,
        "min_raise": 100,
        "blinds": {"big": 50},
        "active_players": 2,
        "board_cards": [],
    })

    assert action == ("raise", 160)


def test_ev_initiative_bot_does_not_open_weak_early_position_preflop():
    bot = EVInitiativeBot()

    action = bot.get_action({
        "hero_equity": 0.04,
        "pot_size": 75,
        "call_amount": 50,
        "stack_size": 5000,
        "min_raise": 100,
        "blinds": {"big": 50},
        "active_players": 9,
        "position": "UTG",
        "board_cards": [],
    })

    assert action == ("fold", 0)


def test_ev_initiative_bot_does_not_check_from_non_blind_preflop_free_state():
    bot = EVInitiativeBot()

    action = bot.get_action({
        "hero_equity": 0.04,
        "pot_size": 75,
        "call_amount": 0,
        "stack_size": 5000,
        "min_raise": 100,
        "blinds": {"big": 50},
        "active_players": 9,
        "position": "UTG",
        "board_cards": [],
    })

    assert action == ("fold", 0)


def test_ev_initiative_bot_can_open_strong_late_position_preflop():
    bot = EVInitiativeBot()

    action = bot.get_action({
        "hero_equity": 0.63,
        "pot_size": 75,
        "call_amount": 50,
        "stack_size": 5000,
        "min_raise": 100,
        "blinds": {"big": 50},
        "active_players": 3,
        "position": "BTN",
        "board_cards": [],
    })

    assert action == ("raise", 135)


def test_ev_initiative_bot_can_check_big_blind_preflop_free_state():
    bot = EVInitiativeBot()

    action = bot.get_action({
        "hero_equity": 0.04,
        "pot_size": 100,
        "call_amount": 0,
        "stack_size": 5000,
        "min_raise": 100,
        "blinds": {"big": 50},
        "active_players": 2,
        "position": "BB",
        "board_cards": [],
    })

    assert action == ("call", 0)


def test_ev_initiative_bot_shoves_short_stack_with_strong_edge():
    bot = EVInitiativeBot()

    action = bot.get_action({
        "hero_equity": 0.70,
        "pot_size": 1000,
        "call_amount": 100,
        "stack_size": 450,
        "min_raise": 100,
        "blinds": {"big": 50},
        "active_players": 2,
        "board_cards": [],
    })

    assert action == ("raise", 350)


def test_ev_initiative_call_margin_can_tighten_marginal_calls():
    bot = EVInitiativeBot(call_margin=0.10)

    action = bot.get_action({
        "hero_equity": 0.52,
        "pot_size": 100,
        "call_amount": 100,
        "stack_size": 5000,
        "min_raise": 100,
        "blinds": {"big": 50},
        "active_players": 2,
        "board_cards": [],
    })

    assert action == ("fold", 0)


def test_ev_initiative_open_threshold_shift_tightens_opens():
    bot = EVInitiativeBot(open_threshold_shift=0.20)

    action = bot.get_action({
        "hero_equity": 0.63,
        "pot_size": 75,
        "call_amount": 50,
        "stack_size": 5000,
        "min_raise": 100,
        "blinds": {"big": 50},
        "active_players": 3,
        "position": "BTN",
        "board_cards": [],
    })

    assert action == ("call", 0)
