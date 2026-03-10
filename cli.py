"""CLI entry point for `fx-engine` console command."""

import uvicorn


def main() -> None:
    """Start the FX Engine API server."""
    uvicorn.run(
        "fx_engine.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
        access_log=True,
    )


if __name__ == "__main__":
    main()
