/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 19, 2024.
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
#if __has_include (<android/dlext_private.h>)
#define IS_ANDROID_DL
#endif

#include "page_size_compat_helpers.h"

#include <android-base/properties.h>

#if defined(IS_ANDROID_DL)
#include <android/dlext_private.h>
#endif

TEST(PageSize16KiBCompatTest, ElfAlignment4KiB_LoadElf) {
  if (getpagesize() != 0x4000) {
    GTEST_SKIP() << "This test is only applicable to 16kB page-size devices";
  }

  bool app_compat_enabled =
      android::base::GetBoolProperty("bionic.linker.16kb.app_compat.enabled", false);
  std::string lib = GetTestLibRoot() + "/libtest_elf_max_page_size_4kib.so";
  void* handle = nullptr;

  OpenTestLibrary(lib, !app_compat_enabled, &handle);

  if (app_compat_enabled) CallTestFunction(handle);
}

TEST(PageSize16KiBCompatTest, ElfAlignment4KiB_LoadElf_perAppOption) {
  if (getpagesize() != 0x4000) {
    GTEST_SKIP() << "This test is only applicable to 16kB page-size devices";
  }

#if defined(IS_ANDROID_DL)
  android_set_16kb_appcompat_mode(true);
#endif

  std::string lib = GetTestLibRoot() + "/libtest_elf_max_page_size_4kib.so";
  void* handle = nullptr;

  OpenTestLibrary(lib, false /*should_fail*/, &handle);
  CallTestFunction(handle);

#if defined(IS_ANDROID_DL)
  android_set_16kb_appcompat_mode(false);
#endif
}
