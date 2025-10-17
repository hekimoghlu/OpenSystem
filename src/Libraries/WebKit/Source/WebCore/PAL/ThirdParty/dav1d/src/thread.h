/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 7, 2023.
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
#ifndef DAV1D_SRC_THREAD_H
#define DAV1D_SRC_THREAD_H

#if defined(_WIN32)

#include <limits.h>
#include <windows.h>

#define PTHREAD_ONCE_INIT INIT_ONCE_STATIC_INIT

typedef struct {
    HANDLE h;
    void *(*func)(void*);
    void *arg;
} pthread_t;

typedef struct {
    unsigned stack_size;
} pthread_attr_t;

typedef SRWLOCK pthread_mutex_t;
typedef CONDITION_VARIABLE pthread_cond_t;
typedef INIT_ONCE pthread_once_t;

void dav1d_init_thread(void);
void dav1d_set_thread_name(const wchar_t *name);
#define dav1d_set_thread_name(name) dav1d_set_thread_name(L##name)

int dav1d_pthread_create(pthread_t *thread, const pthread_attr_t *attr,
                         void *(*func)(void*), void *arg);
int dav1d_pthread_join(pthread_t *thread, void **res);
int dav1d_pthread_once(pthread_once_t *once_control,
                       void (*init_routine)(void));

#define pthread_create dav1d_pthread_create
#define pthread_join(thread, res) dav1d_pthread_join(&(thread), res)
#define pthread_once   dav1d_pthread_once

static inline int pthread_attr_init(pthread_attr_t *const attr) {
    attr->stack_size = 0;
    return 0;
}

static inline int pthread_attr_destroy(pthread_attr_t *const attr) {
    return 0;
}

static inline int pthread_attr_setstacksize(pthread_attr_t *const attr,
                                            const size_t stack_size)
{
    if (stack_size > UINT_MAX) return 1;
    attr->stack_size = (unsigned) stack_size;
    return 0;
}

static inline int pthread_mutex_init(pthread_mutex_t *const mutex,
                                     const void *const attr)
{
    InitializeSRWLock(mutex);
    return 0;
}

static inline int pthread_mutex_destroy(pthread_mutex_t *const mutex) {
    return 0;
}

static inline int pthread_mutex_lock(pthread_mutex_t *const mutex) {
    AcquireSRWLockExclusive(mutex);
    return 0;
}

static inline int pthread_mutex_unlock(pthread_mutex_t *const mutex) {
    ReleaseSRWLockExclusive(mutex);
    return 0;
}

static inline int pthread_cond_init(pthread_cond_t *const cond,
                                    const void *const attr)
{
    InitializeConditionVariable(cond);
    return 0;
}

static inline int pthread_cond_destroy(pthread_cond_t *const cond) {
    return 0;
}

static inline int pthread_cond_wait(pthread_cond_t *const cond,
                                    pthread_mutex_t *const mutex)
{
    return !SleepConditionVariableSRW(cond, mutex, INFINITE, 0);
}

static inline int pthread_cond_signal(pthread_cond_t *const cond) {
    WakeConditionVariable(cond);
    return 0;
}

static inline int pthread_cond_broadcast(pthread_cond_t *const cond) {
    WakeAllConditionVariable(cond);
    return 0;
}

#else

#include <pthread.h>

#define dav1d_init_thread() do {} while (0)

/* Thread naming support */

#ifdef __linux__

#include <sys/prctl.h>

static inline void dav1d_set_thread_name(const char *const name) {
    prctl(PR_SET_NAME, name);
}

#elif defined(__APPLE__)

static inline void dav1d_set_thread_name(const char *const name) {
    pthread_setname_np(name);
}

#elif defined(__DragonFly__) || defined(__FreeBSD__) || defined(__OpenBSD__)

#if defined(__FreeBSD__)
 /* ALIGN from <sys/param.h> conflicts with ALIGN from "common/attributes.h" */
#define _SYS_PARAM_H_
#include <sys/types.h>
#endif
#include <pthread_np.h>

static inline void dav1d_set_thread_name(const char *const name) {
    pthread_set_name_np(pthread_self(), name);
}

#elif defined(__NetBSD__)

static inline void dav1d_set_thread_name(const char *const name) {
    pthread_setname_np(pthread_self(), "%s", (void*)name);
}

#elif defined(__HAIKU__)

#include <os/kernel/OS.h>

static inline void dav1d_set_thread_name(const char *const name) {
    rename_thread(find_thread(NULL), name);
}

#else

#define dav1d_set_thread_name(name) do {} while (0)

#endif

#endif

#endif /* DAV1D_SRC_THREAD_H */
