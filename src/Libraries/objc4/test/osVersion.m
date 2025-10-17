/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 23, 2024.
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

// TEST_CONFIG OS=!exclavekit
// TEST_CFLAGS -framework Foundation

#include "test.h"
#include "mach-o/dyld_priv.h"
#include <Foundation/Foundation.h>

extern uintptr_t objc_debug_realized_class_generation_count;

// Make sure we're testing on the same OS version that libobjc was built for.
int main(int argc __unused, char **argv __unused) {
    const struct mach_header *libobjcHeader = dyld_image_header_containing_address(&objc_debug_realized_class_generation_count);
    // dyld_get_min_os_version tries to normalize to iOS-aligned numbers, which
    // is not what we want. dyld_get_image_versions gives us the raw info we
    // want.
    __block unsigned libobjcVersion = -1;
    dyld_get_image_versions(libobjcHeader, ^(dyld_platform_t platform, uint32_t sdk_version __unused, uint32_t min_version) {
        if (platform == dyld_get_active_platform()) {
            libobjcVersion = min_version;
        }
    });
    unsigned libobjcMajor = libobjcVersion >> 16;
    unsigned libobjcMinor = (libobjcVersion >> 8) & 0xFF;
    testprintf("libobjc minos is %u.%u\n", libobjcMajor, libobjcMinor);

    NSOperatingSystemVersion osVersion = [[NSProcessInfo processInfo] operatingSystemVersion];
    testprintf("OS version is %u.%u\n", (unsigned)osVersion.majorVersion, (unsigned)osVersion.minorVersion);
    testassertequal(libobjcMajor, osVersion.majorVersion);
    testassertequal(libobjcMinor, osVersion.minorVersion);

    succeed(__FILE__);
}
