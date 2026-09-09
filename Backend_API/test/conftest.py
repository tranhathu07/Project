from fastapi.testclient import TestClient
import pytest
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

from app import models, schema
from app.main import app
from app.database import Base, get_db
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.oauth2 import create_access_token
# 1. Kết nối và tạo database test trước tiên nếu chưa có
SQLALCHEMY_DATABASE_URL = "postgresql://postgres:261267@localhost:2618/fastapi_test"


try:
    conn = psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password="261267",
        host="localhost",
        port=2618,
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()
    cursor.execute("CREATE DATABASE fastapi_test;")
    print("Đã tạo database fastapi_test thành công!")
    cursor.close()
    conn.close()
except psycopg2.errors.DuplicateDatabase:
    print("Database fastapi_test đã tồn tại từ trước rồi.")

# 2. Khởi tạo engine và session cho riêng database test
engine = create_engine(SQLALCHEMY_DATABASE_URL)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# 3. Override dependency để app FastAPI dùng database test trong suốt quá trình chạy test
def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

@pytest.fixture
def session():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

@pytest.fixture
def client():

    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()


# 4. Fixture tự động tạo bảng và dọn sạch dữ liệu trước mỗi bài test
@pytest.fixture(autouse=True)
def cleanup_database():
    models.Base.metadata.drop_all(bind=engine)

    # 2. Tạo lại bảng mới tinh khôi
    models.Base.metadata.create_all(bind=engine)

    yield  # Nhường quyền cho bài test chạy

    # 3. Dọn dẹp lại sau khi test xong
    models.Base.metadata.drop_all(bind=engine)


@pytest.fixture
def test_user(client):
    user_data = {"email": "be@gmail.com",
                 "password": "123456"}
    res = client.post("/users/", json = user_data)

    assert res.status_code ==201
    print(res.json())
    new_user = res.json()
    new_user['password'] = user_data['password']

    return new_user
@pytest.fixture
def token(test_user):
    return create_access_token({"user_id": test_user["id"]})
@pytest.fixture
def authorized_client(client,token):
    client.headers = {
        **client.headers,
        "Authorization": f"Bearer {token}"
    }
    return client
@pytest.fixture
def test_posts(test_user, session):
    posts_data = [{
        "title": "first",
        "content" : "first content",
        "owner_id": test_user["id"]
    },
    {"title" : "second",
     "content": "second content",
     "owner_id": test_user["id"]},
     {"title": "third",
      "content":"third content",
        "owner_id":test_user['id']
        }]
    def create_post_model(post):
        return models.Post(**post)
    post_map = map(create_post_model,posts_data)
    posts = list(post_map)
    session.add_all(posts)
    # session.add_all(models.Post(title = "first title", content = "first content", owner_id = test_user['id']),
    #            models.Post(title = "secong", content ="second content" , owner_id = test_user["id"] ),
    #            models.Post(title = "third", content = "third content", owner_id = test_user["id"]))
    session.commit()
    posts = session.query(models.Post).all()
    return posts
