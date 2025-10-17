/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 10, 2023.
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

#include "DebugData.h"
#include "debug_disable.h"
#include "debug_log.h"

pthread_key_t g_disable_key;

bool DebugCallsDisabled() {
  if (g_debug == nullptr || pthread_getspecific(g_disable_key) != nullptr) {
    return true;
  }
  return false;
}

bool DebugDisableInitialize() {
  int error = pthread_key_create(&g_disable_key, nullptr);
  if (error != 0) {
    error_log("pthread_key_create failed: %s", strerror(error));
    return false;
  }
  pthread_setspecific(g_disable_key, nullptr);

  return true;
}

void DebugDisableFinalize() {
  pthread_key_delete(g_disable_key);
}

void DebugDisableSet(bool disable) {
  if (disable) {
    pthread_setspecific(g_disable_key, reinterpret_cast<void*>(1));
  } else {
    pthread_setspecific(g_disable_key, nullptr);
  }
}
