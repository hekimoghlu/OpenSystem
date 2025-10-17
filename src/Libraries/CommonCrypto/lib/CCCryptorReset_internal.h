/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 14, 2022.
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
#ifndef CCCryptorReset_internal_h
#define CCCryptorReset_internal_h

#if defined(_MSC_VER) || defined(__ANDROID__)

#define ProgramLinkedOnOrAfter_macOS1013_iOS11() true

#else

/* WORKAROUND: Manually inserting the requested function */

#include <Availability.h>
#include <TargetConditionals.h>
#include <mach-o/dyld.h>

/* FROM: dyld-852.2 : SRC/include/mach-o/dyld_priv.h */
typedef uint32_t dyld_platform_t;

typedef struct {
    dyld_platform_t platform;
    uint32_t        version;
} dyld_build_version_t;

/* SOURCE: AvailabilityVersions : ran print_dyld_os_versions.rb */
// dyld_fall_2017_os_versions => bridgeos 2.0 / ios 11.0 / macos 10.13 / tvos 11.0 / watchos 4.0
#define dyld_fall_2017_os_versions                      ({ (dyld_build_version_t){0xffffffff, 0x007e10901}; })

extern bool dyld_program_sdk_at_least(dyld_build_version_t version) __API_AVAILABLE(macos(10.14), ios(12.0), watchos(5.0), tvos(12.0));

#define ProgramLinkedOnOrAfter_macOS1013_iOS11() dyld_program_sdk_at_least(dyld_fall_2017_os_versions)

#endif

#endif /* CCCryptorReset_internal_h */
