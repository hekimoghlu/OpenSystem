/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 2, 2024.
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
#include "linker_translate_path.h"
#include "linker_utils.h"

#if defined(__LP64__)
#define APEX_LIB(apex, name) \
  { "/system/lib64/" name, "/apex/" apex "/lib64/" name }
#else
#define APEX_LIB(apex, name) \
  { "/system/lib/" name, "/apex/" apex "/lib/" name }
#endif


// Workaround for dlopen(/system/lib(64)/<soname>) when .so is in /apex. http://b/121248172
/**
 * Translate /system path to /apex path if needed
 * The workaround should work only when targetSdkVersion < 29.
 *
 * param out_name_to_apex pointing to /apex path
 * return true if translation is needed
 */
bool translateSystemPathToApexPath(const char* name, std::string* out_name_to_apex) {
  static constexpr const char* kPathTranslation[][2] = {
      APEX_LIB("com.android.i18n", "libicui18n.so"),
      APEX_LIB("com.android.i18n", "libicuuc.so")
  };

  if (name == nullptr) {
    return false;
  }

  auto comparator = [name](auto p) { return strcmp(name, p[0]) == 0; };

  if (get_application_target_sdk_version() < 29) {
    if (auto it =
            std::find_if(std::begin(kPathTranslation), std::end(kPathTranslation), comparator);
        it != std::end(kPathTranslation)) {
      *out_name_to_apex = (*it)[1];
      return true;
    }
  }

  return false;
}
// End Workaround for dlopen(/system/lib/<soname>) when .so is in /apex.
