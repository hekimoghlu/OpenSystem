/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 11, 2023.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#include "hi_locl.h"
#ifdef HAVE_GCD
#include <dispatch/dispatch.h>
#else
#include "heim_threads.h"
#endif

struct heim_icred {
    uid_t uid;
    gid_t gid;
    pid_t pid;
    pid_t session;
    audit_token_t audit_token;
    /* socket based auth */
    struct sockaddr *client;
    struct sockaddr_storage __clientss;
    krb5_socklen_t client_size;
    struct sockaddr *server;
    struct sockaddr_storage __serverss;
    krb5_socklen_t server_size;

};

void
heim_ipc_free_cred(heim_icred cred)
{
    free(cred);
}

uid_t
heim_ipc_cred_get_uid(heim_icred cred)
{
    return cred->uid;
}

gid_t
heim_ipc_cred_get_gid(heim_icred cred)
{
    return cred->gid;
}

pid_t
heim_ipc_cred_get_pid(heim_icred cred)
{
    return cred->pid;
}

pid_t
heim_ipc_cred_get_session(heim_icred cred)
{
    return cred->session;
}

audit_token_t
heim_ipc_cred_get_audit_token(heim_icred cred)
{
    return cred->audit_token;
}

struct sockaddr *
heim_ipc_cred_get_client_address(heim_icred cred, krb5_socklen_t *sa_size)
{
    *sa_size = cred->client_size;
    return cred->client;
}

struct sockaddr *
heim_ipc_cred_get_server_address(heim_icred cred, krb5_socklen_t *sa_size)
{
    *sa_size = cred->server_size;
    return cred->server;
}

int
_heim_ipc_create_cred(uid_t uid, gid_t gid, pid_t pid, pid_t session, heim_icred *cred)
{
    *cred = calloc(1, sizeof(**cred));
    if (*cred == NULL)
	return ENOMEM;
    (*cred)->uid = uid;
    (*cred)->gid = gid;
    (*cred)->pid = pid;
    (*cred)->session = session;
    return 0;
}

int
_heim_ipc_create_cred_with_audit_token(uid_t uid, gid_t gid, pid_t pid, pid_t session, audit_token_t audit_token, heim_icred *cred)
{
    int res = _heim_ipc_create_cred(uid, gid, pid, session, cred);
    if (res == 0) {
	(*cred)->audit_token = audit_token;
    }
    return 0;
}

int
_heim_ipc_create_network_cred(struct sockaddr *client, krb5_socklen_t client_size,
			      struct sockaddr *server, krb5_socklen_t server_size,
			      heim_icred *cred)
{
    *cred = calloc(1, sizeof(**cred));
    if (*cred == NULL)
	return ENOMEM;
    (*cred)->uid = (uid_t)-1;
    (*cred)->gid = (uid_t)-1;
    (*cred)->pid = (uid_t)-1;
    (*cred)->session = (uid_t)-1;

    if (client_size > sizeof((*cred)->__clientss))
	client_size = sizeof((*cred)->__clientss);
    memcpy(&(*cred)->__clientss, client, client_size);
    (*cred)->client_size = client_size;
    (*cred)->client = (struct sockaddr *)&(*cred)->__clientss;

    if (server_size > sizeof((*cred)->__serverss))
	server_size = sizeof((*cred)->__serverss);
    memcpy(&(*cred)->__serverss, server, server_size);
    (*cred)->server_size = server_size;
    (*cred)->server = (struct sockaddr *)&(*cred)->__serverss;

    return 0;
}

#ifndef HAVE_GCD
struct heim_isemaphore {
    HEIMDAL_MUTEX mutex;
    pthread_cond_t cond;
    long counter;
};
#endif

heim_isemaphore
heim_ipc_semaphore_create(long value)
{
#ifdef HAVE_GCD
    return (heim_isemaphore)dispatch_semaphore_create(value);
#elif !defined(ENABLE_PTHREAD_SUPPORT)
    heim_assert(0, "no semaphore support w/o pthreads");
    return NULL;
#else
    heim_isemaphore s = malloc(sizeof(*s));
    if (s == NULL)
	return NULL;
    HEIMDAL_MUTEX_init(&s->mutex);
    pthread_cond_init(&s->cond, NULL);
    s->counter = value;
    return s;
#endif
}

long
heim_ipc_semaphore_wait(heim_isemaphore s, time_t t)
{
#ifdef HAVE_GCD
    uint64_t timeout;
    if (t == HEIM_IPC_WAIT_FOREVER)
	timeout = DISPATCH_TIME_FOREVER;
    else
	timeout = (uint64_t)t * NSEC_PER_SEC;

    return dispatch_semaphore_wait((dispatch_semaphore_t)s, timeout);
#elif !defined(ENABLE_PTHREAD_SUPPORT)
    heim_assert(0, "no semaphore support w/o pthreads");
    return 0;
#else
    HEIMDAL_MUTEX_lock(&s->mutex);
    /* if counter hits below zero, we get to wait */
    if (--s->counter < 0) {
	int ret;

	if (t == HEIM_IPC_WAIT_FOREVER)
	    ret = pthread_cond_wait(&s->cond, &s->mutex);
	else {
	    struct timespec ts;
	    ts.tv_sec = t;
	    ts.tv_nsec = 0;
	    ret = pthread_cond_timedwait(&s->cond, &s->mutex, &ts);
	}
	if (ret) {
	    HEIMDAL_MUTEX_unlock(&s->mutex);
	    return errno;
	}
    }
    HEIMDAL_MUTEX_unlock(&s->mutex);

    return 0;
#endif
}

long
heim_ipc_semaphore_signal(heim_isemaphore s)
{
#ifdef HAVE_GCD
    return dispatch_semaphore_signal((dispatch_semaphore_t)s);
#elif !defined(ENABLE_PTHREAD_SUPPORT)
    heim_assert(0, "no semaphore support w/o pthreads");
    return EINVAL;
#else
    int wakeup;
    HEIMDAL_MUTEX_lock(&s->mutex);
    wakeup = (++s->counter == 0) ;
    HEIMDAL_MUTEX_unlock(&s->mutex);
    if (wakeup)
	pthread_cond_signal(&s->cond);
    return 0;
#endif
}

void
heim_ipc_semaphore_release(heim_isemaphore s)
{
#ifdef HAVE_GCD
    dispatch_release((dispatch_semaphore_t)s);
#elif !defined(ENABLE_PTHREAD_SUPPORT)
    heim_assert(0, "no semaphore support w/o pthreads");
#else
    HEIMDAL_MUTEX_lock(&s->mutex);
    if (s->counter != 0)
	abort();
    HEIMDAL_MUTEX_unlock(&s->mutex);
    HEIMDAL_MUTEX_destroy(&s->mutex);
    pthread_cond_destroy(&s->cond);
    free(s);
#endif
}

void
heim_ipc_free_data(heim_idata *data)
{
    if (data->data)
	free(data->data);
    data->data = NULL;
    data->length = 0;
}
