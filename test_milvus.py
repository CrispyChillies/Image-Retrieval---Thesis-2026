from pymilvus import MilvusClient
import sys

try:
    client = MilvusClient(
        uri="https://in03-fde920eeaa58fb7.serverless.aws-eu-central-1.cloud.zilliz.com:443",
        token="311672275357fc64ad2f2eebfa9f69c55886a2bb2e001d6966e3d4cbbb235120e0bf4347cb0299295549bb02ab8ef89f201a2053"  # Replace with fresh token
    )
    print("✓ Connection successful!")
    print(f"Collections: {client.list_collections()}")
except Exception as e:
    print(f" ✗ Connection failed: {e}")
    sys.exit(1)