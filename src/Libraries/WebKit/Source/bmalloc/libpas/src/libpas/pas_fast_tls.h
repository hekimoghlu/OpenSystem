/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 16, 2025.
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
#ifndef PAS_FAST_TLS_H
#define PAS_FAST_TLS_H

#include "pas_heap_lock.h"
#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

struct pas_fast_tls;
typedef struct pas_fast_tls pas_fast_tls;

#if PAS_HAVE_PTHREAD_MACHDEP_H

#define PAS_HAVE_THREAD_KEYWORD 0
#define PAS_HAVE_PTHREAD_TLS 0

struct pas_fast_tls {
    bool is_initialized;
};

#define PAS_FAST_TLS_INITIALIZER { .is_initialized = false }

#define PAS_FAST_TLS_CONSTRUCT_IF_NECESSARY(static_key, passed_fast_tls, passed_destructor) do { \
        pas_fast_tls* fast_tls = (passed_fast_tls); \
        void (*the_destructor)(void* arg) = (passed_destructor); \
        pas_heap_lock_assert_held(); \
        if (!fast_tls->is_initialized) { \
            pthread_key_init_np(static_key, the_destructor); \
            fast_tls->is_initialized = true; \
        } \
    } while (false)

#define PAS_FAST_TLS_GET(static_key, fast_tls) \
    _pthread_getspecific_direct(static_key)

#define PAS_FAST_TLS_SET(static_key, fast_tls, value) \
    _pthread_setspecific_direct(static_key, (value))

#else /* PAS_HAVE_PTHREAD_MACHDEP_H -> so !PAS_HAVE_PTHREAD_MACHDEP_H */

struct pas_fast_tls {
    bool is_initialized;
    pthread_key_t key;
};

#define PAS_FAST_TLS_INITIALIZER { .is_initialized = false }

/* This assumes that we will initialize TLS from some thread without racing. That's true, but it's
   not a great assumption. On the other hand, the PAS_HAVE_PTHREAD_MACHDEP_H path makes no such
   assumption.

   The way that the code makes this assumption is that there is no fencing between checking
   is_initialized and using the key. */

#define PAS_FAST_TLS_CONSTRUCT_IF_NECESSARY(static_key, passed_fast_tls, passed_destructor) do { \
        pas_fast_tls* fast_tls = (passed_fast_tls); \
        void (*the_destructor)(void* arg) = (passed_destructor); \
        pas_heap_lock_assert_held(); \
        if (!fast_tls->is_initialized) { \
            pthread_key_create(&fast_tls->key, the_destructor); \
            fast_tls->is_initialized = true; \
        } \
    } while (false)

#if PAS_OS(DARWIN)

#define PAS_HAVE_THREAD_KEYWORD 0
#define PAS_HAVE_PTHREAD_TLS 1

/* __thread keyword implementation does not work since __thread value will be reset to the initial value after it is cleared.
   This broke our pthread exiting detection. We use repeated pthread_setspecific to successfully shutting down. */
#define PAS_FAST_TLS_GET(static_key, passed_fast_tls) ({ \
        pas_fast_tls* fast_tls = (passed_fast_tls); \
        void* result; \
        if (fast_tls->is_initialized) \
            result = pthread_getspecific((fast_tls)->key); \
        else \
            result = NULL; \
        result; \
    })

#define PAS_FAST_TLS_SET(static_key, passed_fast_tls, value) do { \
        pas_fast_tls* fast_tls = (passed_fast_tls); \
        PAS_ASSERT(fast_tls->is_initialized); \
        pthread_setspecific(fast_tls->key, (value)); \
    } while (false)

#else

#define PAS_HAVE_THREAD_KEYWORD 1
#define PAS_HAVE_PTHREAD_TLS 0

/* This is the PAS_HAVE_THREAD_KEYWORD case. Hence, static_key here is not actually a key, but a thread
   local cache of the value (declared with the __thread attribute). Regardless of whether fast_tls has been
   initialized yet or not, it is safe to access this thread local cache of the value. */
#define PAS_FAST_TLS_GET(static_key, fast_tls) static_key

#define PAS_FAST_TLS_SET(static_key, passed_fast_tls, passed_value) do { \
        pas_fast_tls* fast_tls = (passed_fast_tls); \
        PAS_TYPEOF(passed_value) value = (passed_value); \
        PAS_ASSERT(fast_tls->is_initialized); \
        static_key = value; \
        if (((uintptr_t)value) != PAS_THREAD_LOCAL_CACHE_DESTROYED) { \
            /* Using pthread_setspecific to configure callback for thread exit. */ \
            pthread_setspecific(fast_tls->key, value); \
        } \
    } while (false)

#endif

#endif /* PAS_HAVE_PTHREAD_MACHDEP_H -> so end of !PAS_HAVE_PTHREAD_MACHDEP_H */

PAS_END_EXTERN_C;

#endif /* PAS_FAST_TLS_H */

