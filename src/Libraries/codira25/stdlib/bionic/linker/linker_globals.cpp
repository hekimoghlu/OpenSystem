/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 5, 2025.
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
#include "linker.h"
#include "linker_globals.h"
#include "linker_namespaces.h"

#include "android-base/stringprintf.h"

int g_argc = 0;
char** g_argv = nullptr;
char** g_envp = nullptr;

android_namespace_t g_default_namespace;

std::unordered_map<uintptr_t, soinfo*> g_soinfo_handles_map;

platform_properties g_platform_properties;

static char __linker_dl_err_buf[768];

char* linker_get_error_buffer() {
  return &__linker_dl_err_buf[0];
}

size_t linker_get_error_buffer_size() {
  return sizeof(__linker_dl_err_buf);
}

bool DL_ERROR_AFTER(int target_sdk_version, const char* fmt, ...) {
  std::string result;
  va_list ap;
  va_start(ap, fmt);
  android::base::StringAppendV(&result, fmt, ap);
  va_end(ap);

  if (get_application_target_sdk_version() < target_sdk_version) {
    android::base::StringAppendF(&result,
                                 " and will not work when the app moves to "
                                 "targetSdkVersion %d or later "
                                 "(see https://android.googlesource.com/platform/bionic/+/main/"
                                 "android-changes-for-ndk-developers.md); "
                                 "allowing for now because this app's "
                                 "targetSdkVersion is still %d",
                                 target_sdk_version,
                                 get_application_target_sdk_version());
    DL_WARN("Warning: %s", result.c_str());
    return false;
  }
  DL_ERR_AND_LOG("%s", result.c_str());
  return true;
}
