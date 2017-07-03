import uuid

import numpy as np
from tinydb import TinyDB

db = TinyDB('database.json')
table = db.table('users')


def distance(a, b):
    d = b - a
    return np.dot(d, d)


def create_user(vector):
    vectors = list([list(vector)])
    user = {'uid': str(uuid.uuid1()), 'vectors': vectors}
    table.insert(user)
    return user


# TODO: KDThree
def find_user(vector):
    for user in table.all():
        vectors = user['vectors']

        dst = 0.0
        for v in vectors:
            dst += distance(vector, v)

        dst /= len(vectors)

        if dst < 1.0:
            vectors.append(list(vector))
            table.update({'vectors': list(vectors)}, eids=[user.eid])

            return user
