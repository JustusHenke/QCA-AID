"""
Regressionstest: Die Ausgabe der Mehrfachkodierungs-Statistiken darf einen
bereits abgeschlossenen Analyse-Lauf nicht als Fehler beenden.

Hintergrund (v0.13.6): `calculate_multiple_coding_stats` lieff nach Export und
Review und stuerzte bei `category = None` mit
`TypeError: '<' not supported between instances of 'NoneType' and 'str'` ab.
Der Lauf endete dadurch mit "Fehler in der Hauptausfuehrung", obwohl alle
Ergebnisse geschrieben waren.

`QCA_AID_assets.main._report_multiple_coding_stats()` kapselt den Aufruf: bei
Fehlern gibt es eine WARNING ins Log plus einen kurzen Konsolenhinweis - ohne
Exception nach oben.
"""

import logging

import QCA_AID_assets.main as main_module


def test_stats_error_is_logged_but_not_raised(monkeypatch, caplog, capsys):
    def _boom(codings):
        raise TypeError(
            "'<' not supported between instances of 'NoneType' and 'str'"
        )

    monkeypatch.setattr(main_module, "calculate_multiple_coding_stats", _boom)

    with caplog.at_level(logging.WARNING, logger=main_module.__name__):
        # Darf nicht werfen
        main_module._report_multiple_coding_stats(
            [{"segment_id": "doc_chunk_0", "category": None}]
        )

    out = capsys.readouterr().out
    assert "[WARNUNG]" in out
    assert "[HINWEIS]" in out
    assert "abgeschlossen" in out
    # Traceback landet im Log
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, "Erwartet wurde eine WARNING im Log"
    assert "Mehrfachkodierungs-Statistiken" in warnings[0].getMessage()


def test_success_path_does_not_print_warning(capsys):
    main_module._report_multiple_coding_stats(
        [
            {"segment_id": "doc_chunk_0", "coder_id": "auto_1", "category": "A"},
            {"segment_id": "doc_chunk_0", "coder_id": "auto_2", "category": "B"},
        ]
    )

    out = capsys.readouterr().out
    assert "[WARNUNG]" not in out


def test_real_none_category_codings_do_not_warn(capsys):
    """Der urspruengliche Fehlerfall darf jetzt gar nicht mehr auftreten."""
    main_module._report_multiple_coding_stats(
        [
            {"segment_id": "doc_chunk_0", "coder_id": "auto_1", "category": None},
            {"segment_id": "doc_chunk_0", "coder_id": "auto_2", "category": "Alpha"},
        ]
    )

    out = capsys.readouterr().out
    assert "[WARNUNG]" not in out
