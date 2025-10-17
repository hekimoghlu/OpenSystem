/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 28, 2024.
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
#import <Foundation/Foundation.h>
#include <dlfcn.h>
#include <mach-o/dyld.h>
#include <stdint.h>
#include <stdlib.h>

#include "rtc_base/checks.h"

namespace webrtc {
namespace test {

void GetNSExecutablePath(std::string* path) {
  RTC_DCHECK(path);
  // Executable path can have relative references ("..") depending on
  // how the app was launched.
  uint32_t executable_length = 0;
  _NSGetExecutablePath(NULL, &executable_length);
  RTC_DCHECK_GT(executable_length, 1u);
  char executable_path[PATH_MAX + 1];
  int rv = _NSGetExecutablePath(executable_path, &executable_length);
  RTC_DCHECK_EQ(rv, 0);

  char full_path[PATH_MAX];
  if (realpath(executable_path, full_path) == nullptr) {
    *path = "";
    return;
  }

  *path = full_path;
}

}  // namespace test
}  // namespace webrtc
