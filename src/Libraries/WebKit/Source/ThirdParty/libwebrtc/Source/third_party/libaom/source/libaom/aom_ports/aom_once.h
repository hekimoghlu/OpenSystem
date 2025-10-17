/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 24, 2025.
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
#ifndef AOM_AOM_PORTS_AOM_ONCE_H_
#define AOM_AOM_PORTS_AOM_ONCE_H_

#include "config/aom_config.h"

/* Implement a function wrapper to guarantee initialization
 * thread-safety for library singletons.
 *
 * NOTE: This function uses static locks, and can only be
 * used with one common argument per compilation unit. So
 *
 * file1.c:
 *   aom_once(foo);
 *   ...
 *   aom_once(foo);
 *
 * file2.c:
 *   aom_once(bar);
 *
 * will ensure foo() and bar() are each called only once, but in
 *
 * file1.c:
 *   aom_once(foo);
 *   aom_once(bar):
 *
 * bar() will never be called because the lock is used up
 * by the call to foo().
 */

#if CONFIG_MULTITHREAD && defined(_WIN32)
#undef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
/* Declare a per-compilation-unit state variable to track the progress
 * of calling func() only once. This must be at global scope because
 * local initializers are not thread-safe in MSVC prior to Visual
 * Studio 2015.
 */
static INIT_ONCE aom_init_once = INIT_ONCE_STATIC_INIT;

static void aom_once(void (*func)(void)) {
  BOOL pending;
  InitOnceBeginInitialize(&aom_init_once, 0, &pending, NULL);
  if (!pending) {
    // Initialization has already completed.
    return;
  }
  func();
  InitOnceComplete(&aom_init_once, 0, NULL);
}

#elif CONFIG_MULTITHREAD && HAVE_PTHREAD_H
#include <pthread.h>
static void aom_once(void (*func)(void)) {
  static pthread_once_t lock = PTHREAD_ONCE_INIT;
  pthread_once(&lock, func);
}

#else
/* Default version that performs no synchronization. */

static void aom_once(void (*func)(void)) {
  static volatile int done;

  if (!done) {
    func();
    done = 1;
  }
}
#endif

#endif  // AOM_AOM_PORTS_AOM_ONCE_H_
