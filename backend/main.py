from fastapi import FastAPI
from middlewares import add_cors_middleware
from routers import document_parser, logs

app = FastAPI()

# Add CORS
add_cors_middleware(app)

# Include routers
app.include_router(document_parser.router)
app.include_router(logs.router)
