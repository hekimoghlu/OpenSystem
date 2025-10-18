# Open Desktop Ratings Service

A Flask web service for submitting application reviews

## Setting up local environment

The easiest way is to run `docker compose up` in the root directory. This will
bring up a local ODRS instance with all needed services. Gunicorn can be
directly accessed at http://localhost:8080, while nginx that would be used for
production traffic listens at http://localhost:8000.

The entrypoint script creates a default admin user with login `admin@test.com`
and password `Pa$$w0rd`.

### Example local requests

Submit a review:
```shell
curl -w '\n' -X POST http://localhost:8080/1.0/reviews/api/submit --json '{"app_id": "org.example.app", "locale": "en_US", "summary": "Good App", "description": "Loved it", "user_hash": "a17fed27eaa842282862ff7c1b9c8395a26ac322", "version": "1.0", "distro": "debian", "rating": 100, "user_display": "Happy User"}'
```

Query ratings:
```shell
curl -w '\n' http://localhost:8080/1.0/reviews/api/ratings/org.example.app
```

### Generating migration files

If you modify the database models, you'll need to generate migration files.

```shell
env 'SQLALCHEMY_DATABASE_URI=mysql+mysqldb://odrs:odrspasswd@127.0.0.1:3306/odrs?charset=utf8mb4' flask --app odrs/ db migrate
```

This will drop a new file in `migrations/versions/`.
Rename it to add a little description.

## Deployment

New commits to the master branch are automatically deployed to the testing
instance at https://odrs-dev.apps.openshift4.gnome.org/.

A commit can be promoted to production environment by manually starting the
`odrs` job [here](https://gitlab.gnome.org/Infrastructure/odrs-web/-/jobs).

## I have a question

Email me or grab me on IRC (`hughsie@libera.chat`).
