from app import schema
import pytest
from jose import jwt
from app.config import settings




def test_create_user(client):
    res = client.post(
        "/users/", json={"email": "hello@gmail.com", "password": "123456"}
    )
    new_user = schema.UserOut(**res.json())
    assert res.status_code == 201
    assert new_user.email == "hello@gmail.com"

def test_login_user(client,test_user):
    res = client.post("/login" , data = {"username":test_user["email"], "password":test_user["password"]})
    login_res = schema.Token(**res.json())
    payload = jwt.decode(login_res.access_token, settings.secret_key, algorithms= settings.algorithm)
    id: str = payload.get("user_id")
    assert id == test_user['id']
    assert login_res.token_type == 'bearer'
    assert res.status_code == 200

@pytest.mark.parametrize("email,password,status_code", [
    ('wrongemail@gmail.com', '123',403),
    ('be@gmai.com', 'wrongpass',403),
    ('wrongemail@gmail.com','wrongpass',403),
    (None, '123456',422),
    ('be@gmail.com',None,422)
])
def test_incorrect_login(test_user, client, email,password,status_code):
    res = client.post("/login", data = {"username": email, "password": password})
    assert res.status_code == status_code
    