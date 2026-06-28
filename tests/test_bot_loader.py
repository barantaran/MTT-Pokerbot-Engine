from main import load_bots


def test_load_bots_includes_icm_tight_bot():
    bot_names = {bot.__class__.__name__ for bot in load_bots()}

    assert "ICMTightBot" in bot_names
    assert "TournamentICMEquityBot" in bot_names
    assert "AdaptiveTournamentICMEquityBot" in bot_names
    assert "ButtonStealTournamentICMEquityBot" in bot_names
