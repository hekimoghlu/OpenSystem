/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 17, 2022.
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
#ifndef LIBUSB_THREADS_WINDOWS_H
#define LIBUSB_THREADS_WINDOWS_H

#define WINAPI_CHECK(expression)	ASSERT_NE(expression, 0)

#define USBI_MUTEX_INITIALIZER	0L
typedef LONG usbi_mutex_static_t;
static inline void usbi_mutex_static_lock(usbi_mutex_static_t *mutex)
{
	while (InterlockedExchange(mutex, 1L) == 1L)
		SleepEx(0, TRUE);
}
static inline void usbi_mutex_static_unlock(usbi_mutex_static_t *mutex)
{
	InterlockedExchange(mutex, 0L);
}

typedef CRITICAL_SECTION usbi_mutex_t;
static inline void usbi_mutex_init(usbi_mutex_t *mutex)
{
	InitializeCriticalSection(mutex);
}
static inline void usbi_mutex_lock(usbi_mutex_t *mutex)
{
	EnterCriticalSection(mutex);
}
static inline void usbi_mutex_unlock(usbi_mutex_t *mutex)
{
	LeaveCriticalSection(mutex);
}
static inline int usbi_mutex_trylock(usbi_mutex_t *mutex)
{
	return TryEnterCriticalSection(mutex) != 0;
}
static inline void usbi_mutex_destroy(usbi_mutex_t *mutex)
{
	DeleteCriticalSection(mutex);
}

#if !defined(HAVE_STRUCT_TIMESPEC) && !defined(_TIMESPEC_DEFINED)
#define HAVE_STRUCT_TIMESPEC 1
#define _TIMESPEC_DEFINED 1
struct timespec {
	long tv_sec;
	long tv_nsec;
};
#endif /* HAVE_STRUCT_TIMESPEC || _TIMESPEC_DEFINED */

typedef CONDITION_VARIABLE usbi_cond_t;
static inline void usbi_cond_init(usbi_cond_t *cond)
{
	InitializeConditionVariable(cond);
}
static inline void usbi_cond_wait(usbi_cond_t *cond, usbi_mutex_t *mutex)
{
	WINAPI_CHECK(SleepConditionVariableCS(cond, mutex, INFINITE));
}
int usbi_cond_timedwait(usbi_cond_t *cond,
	usbi_mutex_t *mutex, const struct timeval *tv);
static inline void usbi_cond_broadcast(usbi_cond_t *cond)
{
	WakeAllConditionVariable(cond);
}
static inline void usbi_cond_destroy(usbi_cond_t *cond)
{
	UNUSED(cond);
}

typedef DWORD usbi_tls_key_t;
static inline void usbi_tls_key_create(usbi_tls_key_t *key)
{
	*key = TlsAlloc();
	assert(*key != TLS_OUT_OF_INDEXES);
}
static inline void *usbi_tls_key_get(usbi_tls_key_t key)
{
	return TlsGetValue(key);
}
static inline void usbi_tls_key_set(usbi_tls_key_t key, void *ptr)
{
	WINAPI_CHECK(TlsSetValue(key, ptr));
}
static inline void usbi_tls_key_delete(usbi_tls_key_t key)
{
	WINAPI_CHECK(TlsFree(key));
}

static inline unsigned int usbi_get_tid(void)
{
	return (unsigned int)GetCurrentThreadId();
}

#endif /* LIBUSB_THREADS_WINDOWS_H */
