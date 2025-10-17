/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 4, 2022.
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
#include "private/__bionic_get_shell_path.h"

#include <unistd.h>

const char* __bionic_get_shell_path() {
  // Since API level 28 there's a /bin -> /system/bin symlink that means
  // /bin/sh will work for the device too, but as long as the NDK supports
  // earlier API levels, falling back to /system/bin/sh ensures that static
  // binaries run on those OS versions too.
  // This whole function can be removed and replaced by hard-coded /bin/sh
  // when we no longer support anything below API level 28.
  static bool have_bin_sh = !access("/bin/sh", F_OK);
  return have_bin_sh ? "/bin/sh" : "/system/bin/sh";
}
