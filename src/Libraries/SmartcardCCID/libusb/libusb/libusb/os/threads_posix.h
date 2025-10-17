/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 24, 2024.
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
#ifndef LIBUSB_THREADS_POSIX_H
#define LIBUSB_THREADS_POSIX_H

#include <pthread.h>

#define PTHREAD_CHECK(expression)	ASSERT_EQ(expression, 0)

#define USBI_MUTEX_INITIALIZER	PTHREAD_MUTEX_INITIALIZER
typedef pthread_mutex_t usbi_mutex_static_t;
static inline void usbi_mutex_static_lock(usbi_mutex_static_t *mutex)
{
	PTHREAD_CHECK(pthread_mutex_lock(mutex));
}
static inline void usbi_mutex_static_unlock(usbi_mutex_static_t *mutex)
{
	PTHREAD_CHECK(pthread_mutex_unlock(mutex));
}

typedef pthread_mutex_t usbi_mutex_t;
static inline void usbi_mutex_init(usbi_mutex_t *mutex)
{
	PTHREAD_CHECK(pthread_mutex_init(mutex, NULL));
}
static inline void usbi_mutex_lock(usbi_mutex_t *mutex)
{
	PTHREAD_CHECK(pthread_mutex_lock(mutex));
}
static inline void usbi_mutex_unlock(usbi_mutex_t *mutex)
{
	PTHREAD_CHECK(pthread_mutex_unlock(mutex));
}
static inline int usbi_mutex_trylock(usbi_mutex_t *mutex)
{
	return pthread_mutex_trylock(mutex) == 0;
}
static inline void usbi_mutex_destroy(usbi_mutex_t *mutex)
{
	PTHREAD_CHECK(pthread_mutex_destroy(mutex));
}

typedef pthread_cond_t usbi_cond_t;
void usbi_cond_init(pthread_cond_t *cond);
static inline void usbi_cond_wait(usbi_cond_t *cond, usbi_mutex_t *mutex)
{
	PTHREAD_CHECK(pthread_cond_wait(cond, mutex));
}
int usbi_cond_timedwait(usbi_cond_t *cond,
	usbi_mutex_t *mutex, const struct timeval *tv);
static inline void usbi_cond_broadcast(usbi_cond_t *cond)
{
	PTHREAD_CHECK(pthread_cond_broadcast(cond));
}
static inline void usbi_cond_destroy(usbi_cond_t *cond)
{
	PTHREAD_CHECK(pthread_cond_destroy(cond));
}

typedef pthread_key_t usbi_tls_key_t;
static inline void usbi_tls_key_create(usbi_tls_key_t *key)
{
	PTHREAD_CHECK(pthread_key_create(key, NULL));
}
static inline void *usbi_tls_key_get(usbi_tls_key_t key)
{
	return pthread_getspecific(key);
}
static inline void usbi_tls_key_set(usbi_tls_key_t key, void *ptr)
{
	PTHREAD_CHECK(pthread_setspecific(key, ptr));
}
static inline void usbi_tls_key_delete(usbi_tls_key_t key)
{
	PTHREAD_CHECK(pthread_key_delete(key));
}

unsigned int usbi_get_tid(void);

#endif /* LIBUSB_THREADS_POSIX_H */
