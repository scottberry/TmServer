from auth import jwt

from gc3pie import GC3Pie
gc3pie = GC3Pie()

from flask_uwsgi_websocket import GeventWebSocket
websocket = GeventWebSocket()

from flask_redis import FlaskRedis
redis_store = FlaskRedis()

