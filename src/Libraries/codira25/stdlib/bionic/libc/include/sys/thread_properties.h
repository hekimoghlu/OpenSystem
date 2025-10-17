/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 19, 2024.
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
#pragma once

/**
 * @file thread_properties.h
 * @brief Thread properties API.
 *
 * https://sourceware.org/glibc/wiki/ThreadPropertiesAPI
 * API for querying various properties of the current thread, used mostly by
 * the sanitizers.
 *
 * Available since API level 31.
 *
 */

#include <sys/cdefs.h>
#include <unistd.h>

__BEGIN_DECLS

/**
 * Gets the bounds of static TLS for the current thread.
 *
 * Available since API level 31.
 */

#if __BIONIC_AVAILABILITY_GUARD(31)
void __libc_get_static_tls_bounds(void* _Nonnull * _Nonnull __static_tls_begin,
                                  void* _Nonnull * _Nonnull __static_tls_end) __INTRODUCED_IN(31);


/**
 * Registers callback to be called right before the thread is dead.
 * The callbacks are chained, they are called in the order opposite to the order
 * they were registered.
 *
 * The callbacks must be registered only before any threads were created.
 * No signals may arrive during the calls to these callbacks.
 * The callbacks may not access the thread's dynamic TLS because they will have
 * been freed by the time these callbacks are invoked.
 *
 * Available since API level 31.
 */
void __libc_register_thread_exit_callback(void (* _Nonnull __cb)(void)) __INTRODUCED_IN(31);

/**
 * Iterates over all dynamic TLS chunks for the given thread.
 * The thread should have been suspended. It is undefined-behaviour if there is concurrent
 * modification of the target thread's dynamic TLS.
 *
 * Available since API level 31.
 */
void __libc_iterate_dynamic_tls(pid_t __tid,
                                void (* _Nonnull __cb)(void* _Nonnull __dynamic_tls_begin,
                                             void* _Nonnull __dynamic_tls_end,
                                             size_t __dso_id,
                                             void* _Nullable __arg),
                                void* _Nullable __arg) __INTRODUCED_IN(31);

/**
 * Register on_creation and on_destruction callbacks, which will be called after a dynamic
 * TLS creation and before a dynamic TLS destruction, respectively.
 *
 * Available since API level 31.
 */
void __libc_register_dynamic_tls_listeners(
    void (* _Nonnull __on_creation)(void* _Nonnull __dynamic_tls_begin,
                          void* _Nonnull __dynamic_tls_end),
    void (* _Nonnull __on_destruction)(void* _Nonnull __dynamic_tls_begin,
                             void* _Nonnull __dynamic_tls_end)) __INTRODUCED_IN(31);
#endif /* __BIONIC_AVAILABILITY_GUARD(31) */


__END_DECLS
