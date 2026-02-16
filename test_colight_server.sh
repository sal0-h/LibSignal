python -c "
try:
    from agent.colight import CoLightAgent
    print(\"Direct CoLight import: SUCCESS\")
except Exception as e:
    print(\"Direct CoLight import: FAILED\")
    print(\"Error type:\", type(e).__name__)
    print(\"Error message:\", str(e))
    import traceback
    traceback.print_exc()
"
