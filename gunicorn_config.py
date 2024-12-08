workers = 1  # Use single worker to avoid threading issues
worker_class = 'sync'
timeout = 120  # Increase timeout for processing
max_requests = 1  # Restart worker after each request
max_requests_jitter = 5