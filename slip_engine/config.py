import os

class EngineConfig:
    # Supabase Configuration
    SUPABASE_URL: str = os.getenv(
        "SUPABASE_URL", "https://grydnpgufybxvmovdumg.supabase.co"
    )
    
    # Try Kaggle Secrets, then environment variables, then placeholder
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "")
    if not SUPABASE_KEY:
        try:
            from kaggle_secrets import UserSecretsClient
            user_secrets = UserSecretsClient()
            SUPABASE_KEY = user_secrets.get_secret("SUPABASE_SERVICE_KEY")
        except Exception:
            SUPABASE_KEY = "sb_publishable_VXnufLFzFqjaLkNP_u2lbg_kYr3oTfi"

    # Compute & Model Configuration (Kaggle CPU Optimized)
    MODEL_NAME: str = os.getenv("MODEL_NAME", "BAAI/bge-base-en-v1.5")
    EMBEDDING_DIM: int = 768
    BATCH_SIZE: int = 32
    MAX_SESSION_HOURS: float = 11.5  # Safe termination before Kaggle 12h cutoff
    CYCLE_INTERVAL_SECONDS: int = 300  # Poll every 5 minutes
