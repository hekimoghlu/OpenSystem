/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 18, 2023.
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
#include <TargetConditionals.h>
#include <mach-o/dyld.h>
#include <mach-o/dyld_priv.h>
#include "crt_externs.h" /* _NSGetMachExecuteHeader() */

#include "stdio_init.h"

#ifndef PR_96211868_CHECK
#define PR_96211868_CHECK TARGET_OS_OSX
#endif

__attribute__ ((visibility ("hidden")))
bool __ftell_conformance_fix = true;

#if PR_96211868_CHECK
static bool
__chk_ftell_skip_conformance(const struct mach_header *mh) {
  return (dyld_get_active_platform() == PLATFORM_MACOS &&
      !dyld_sdk_at_least(mh, dyld_platform_version_macOS_13_0));
}
#endif

/* Initializer for libc stdio */
__attribute__ ((visibility ("hidden")))
void __stdio_init(void) {
#if PR_96211868_CHECK
    const struct mach_header *hdr = (struct mach_header *)_NSGetMachExecuteHeader();

    if (__chk_ftell_skip_conformance(hdr)) {
        __ftell_conformance_fix = false;
    }
#endif /* PR_96211868_CHECK */
}