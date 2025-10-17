/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 12, 2024.
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
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>

#include "private/bionic_defs.h"

extern "C" void __cxa_finalize(void* dso_handle);
extern "C" void __cxa_thread_finalize();

static pthread_mutex_t g_exit_mutex = PTHREAD_RECURSIVE_MUTEX_INITIALIZER_NP;

__BIONIC_WEAK_FOR_NATIVE_BRIDGE
void exit(int status) {
  // https://austingroupbugs.net/view.php?id=1845
  pthread_mutex_lock(&g_exit_mutex);

  __cxa_thread_finalize();
  __cxa_finalize(nullptr);
  _exit(status);
}
