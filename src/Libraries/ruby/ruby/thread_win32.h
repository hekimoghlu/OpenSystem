/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 25, 2024.
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
/* interface */
#ifndef RUBY_THREAD_WIN32_H
#define RUBY_THREAD_WIN32_H

# ifdef __CYGWIN__
# undef _WIN32
# endif

WINBASEAPI BOOL WINAPI
TryEnterCriticalSection(IN OUT LPCRITICAL_SECTION lpCriticalSection);

typedef struct rb_thread_cond_struct {
    struct cond_event_entry *next;
    struct cond_event_entry *prev;
} rb_nativethread_cond_t;

typedef struct native_thread_data_struct {
    HANDLE interrupt_event;
} native_thread_data_t;

typedef struct rb_global_vm_lock_struct {
    HANDLE lock;
} rb_global_vm_lock_t;

#endif /* RUBY_THREAD_WIN32_H */

