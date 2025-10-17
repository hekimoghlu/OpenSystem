/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 19, 2022.
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
#ifndef RUBY_THREAD_NATIVE_H
#define RUBY_THREAD_NATIVE_H 1

/*
 * This file contains wrapper APIs for native thread primitives
 * which Ruby interpreter uses.
 *
 * Now, we only suppors pthread and Windows threads.
 *
 * If you want to use Ruby's Mutex and so on to synchronize Ruby Threads,
 * please use Mutex directly.
 */


#if defined(_WIN32)
#include <windows.h>
typedef HANDLE rb_nativethread_id_t;

typedef union rb_thread_lock_union {
    HANDLE mutex;
    CRITICAL_SECTION crit;
} rb_nativethread_lock_t;

#elif defined(HAVE_PTHREAD_H)
#include <pthread.h>
typedef pthread_t rb_nativethread_id_t;
typedef pthread_mutex_t rb_nativethread_lock_t;

#else
#error "unsupported thread type"

#endif

RUBY_SYMBOL_EXPORT_BEGIN

rb_nativethread_id_t rb_nativethread_self();

void rb_nativethread_lock_initialize(rb_nativethread_lock_t *lock);
void rb_nativethread_lock_destroy(rb_nativethread_lock_t *lock);
void rb_nativethread_lock_lock(rb_nativethread_lock_t *lock);
void rb_nativethread_lock_unlock(rb_nativethread_lock_t *lock);

RUBY_SYMBOL_EXPORT_END

#endif
