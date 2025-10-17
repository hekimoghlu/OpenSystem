/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 7, 2025.
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
#include <atomic>

#include <android/api-level.h>
#include <android/fdsan.h>

#include "private/bionic_globals.h"

#include "linker.h"

static std::atomic<int> g_target_sdk_version(__ANDROID_API__);

void set_application_target_sdk_version(int target) {
  // translate current sdk_version to platform sdk_version
  if (target == 0) {
    target = __ANDROID_API__;
  }
  g_target_sdk_version = target;

  if (target < 30) {
    android_fdsan_set_error_level_from_property(ANDROID_FDSAN_ERROR_LEVEL_WARN_ONCE);
  }
  if (__libc_shared_globals()->set_target_sdk_version_hook) {
    __libc_shared_globals()->set_target_sdk_version_hook(target);
  }
}

int get_application_target_sdk_version() {
  return g_target_sdk_version;
}

