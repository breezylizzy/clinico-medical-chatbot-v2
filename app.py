from flask import Flask
from app.routes import routes
from app.config import Config

def create_app():
    app = Flask(
        __name__,
        template_folder="app/templates",
        static_folder="app/static"
    )

    app.config.from_object(Config)
    app.register_blueprint(routes)

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, use_reloader=False)
