/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 23, 2023.
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
#include "xmlversionInternal.h"

#ifdef __APPLE__
#include <dispatch/dispatch.h>
#include <mach-o/dyld_priv.h>
#endif

bool linkedOnOrAfterFall2022OSVersions(void)
{
#ifdef FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION
    return true;
#elif defined(LIBXML_LINKED_ON_OR_AFTER_MACOS13_IOS16_WATCHOS9_TVOS16)
    static bool result;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        result = dyld_program_minos_at_least(dyld_fall_2022_os_versions);
    });
    return result;
#else
    return false;
#endif
}

bool linkedOnOrAfter2024EReleases(void)
{
#ifdef LIBXML_LINKED_ON_OR_AFTER_MACOS15_4_IOS18_4_WATCHOS11_4_TVOS18_4_VISIONOS2_4
    static bool result;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        result = dyld_program_sdk_at_least(dyld_2024_SU_E_os_versions);
    });
    return result;
#else
    return false;
#endif
}
