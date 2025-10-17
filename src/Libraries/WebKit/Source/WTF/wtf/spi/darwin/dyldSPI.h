/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 17, 2023.
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
#pragma once

#if USE(APPLE_INTERNAL_SDK)

#include <mach-o/dyld_priv.h>

#ifndef DYLD_IOS_VERSION_11_0
#define DYLD_IOS_VERSION_11_0 0x000B0000
#endif

#ifndef DYLD_IOS_VERSION_11_3
#define DYLD_IOS_VERSION_11_3 0x000B0300
#endif

#ifndef DYLD_IOS_VERSION_12_0
#define DYLD_IOS_VERSION_12_0 0x000C0000
#endif

#ifndef DYLD_IOS_VERSION_12_2
#define DYLD_IOS_VERSION_12_2 0x000C0200
#endif

#ifndef DYLD_IOS_VERSION_13_0
#define DYLD_IOS_VERSION_13_0 0x000D0000
#endif

#ifndef DYLD_IOS_VERSION_13_2
#define DYLD_IOS_VERSION_13_2 0x000D0200
#endif

#ifndef DYLD_IOS_VERSION_13_4
#define DYLD_IOS_VERSION_13_4 0x000D0400
#endif

#ifndef DYLD_IOS_VERSION_14_0
#define DYLD_IOS_VERSION_14_0 0x000E0000
#endif

#ifndef DYLD_IOS_VERSION_14_2
#define DYLD_IOS_VERSION_14_2 0x000E0200
#endif

#ifndef DYLD_IOS_VERSION_14_5
#define DYLD_IOS_VERSION_14_5 0x000E0500
#endif

#ifndef DYLD_IOS_VERSION_15_0
#define DYLD_IOS_VERSION_15_0 0x000f0000
#endif

#ifndef DYLD_IOS_VERSION_15_4
#define DYLD_IOS_VERSION_15_4 0x000f0400
#endif

#ifndef DYLD_IOS_VERSION_16_0
#define DYLD_IOS_VERSION_16_0 0x00100000
#endif

#ifndef DYLD_IOS_VERSION_16_4
#define DYLD_IOS_VERSION_16_4 0x00100400
#endif

#ifndef DYLD_IOS_VERSION_17_0
#define DYLD_IOS_VERSION_17_0 0x00110000
#endif

#ifndef DYLD_IOS_VERSION_17_2
#define DYLD_IOS_VERSION_17_2 0x00110200
#endif

#ifndef DYLD_IOS_VERSION_17_4
#define DYLD_IOS_VERSION_17_4 0x00110400
#endif

#ifndef DYLD_IOS_VERSION_18_0
#define DYLD_IOS_VERSION_18_0 0x00120000
#endif

#ifndef DYLD_MACOSX_VERSION_10_13
#define DYLD_MACOSX_VERSION_10_13 0x000A0D00
#endif

#ifndef DYLD_MACOSX_VERSION_10_14
#define DYLD_MACOSX_VERSION_10_14 0x000A0E00
#endif

#ifndef DYLD_MACOSX_VERSION_10_15
#define DYLD_MACOSX_VERSION_10_15 0x000A0F00
#endif

#ifndef DYLD_MACOSX_VERSION_10_15_1
#define DYLD_MACOSX_VERSION_10_15_1 0x000A0F01
#endif

#ifndef DYLD_MACOSX_VERSION_10_15_4
#define DYLD_MACOSX_VERSION_10_15_4 0x000A0F04
#endif

#ifndef DYLD_MACOSX_VERSION_10_16
#define DYLD_MACOSX_VERSION_10_16 0x000A1000
#endif

#ifndef DYLD_MACOSX_VERSION_11_3
#define DYLD_MACOSX_VERSION_11_3 0x000B0300
#endif

#ifndef DYLD_MACOSX_VERSION_12_00
#define DYLD_MACOSX_VERSION_12_00 0x000c0000
#endif

#ifndef DYLD_MACOSX_VERSION_12_3
#define DYLD_MACOSX_VERSION_12_3 0x000c0300
#endif

#ifndef DYLD_MACOSX_VERSION_13_0
#define DYLD_MACOSX_VERSION_13_0 0x000d0000
#endif

#ifndef DYLD_MACOSX_VERSION_13_3
#define DYLD_MACOSX_VERSION_13_3 0x000d0300
#endif

#ifndef DYLD_MACOSX_VERSION_14_0
#define DYLD_MACOSX_VERSION_14_0 0x000e0000
#endif

#ifndef DYLD_MACOSX_VERSION_14_2
#define DYLD_MACOSX_VERSION_14_2 0x000e0200
#endif

#ifndef DYLD_MACOSX_VERSION_14_4
#define DYLD_MACOSX_VERSION_14_4 0x000e0400
#endif

#ifndef DYLD_MACOSX_VERSION_15_0
#define DYLD_MACOSX_VERSION_15_0 0x000f0000
#endif

#else

typedef uint32_t dyld_platform_t;

typedef struct {
    dyld_platform_t platform;
    uint32_t version;
} dyld_build_version_t;

#define DYLD_IOS_VERSION_3_0 0x00030000
#define DYLD_IOS_VERSION_4_2 0x00040200
#define DYLD_IOS_VERSION_5_0 0x00050000
#define DYLD_IOS_VERSION_6_0 0x00060000
#define DYLD_IOS_VERSION_7_0 0x00070000
#define DYLD_IOS_VERSION_9_0 0x00090000
#define DYLD_IOS_VERSION_10_0 0x000A0000
#define DYLD_IOS_VERSION_11_0 0x000B0000
#define DYLD_IOS_VERSION_11_3 0x000B0300
#define DYLD_IOS_VERSION_12_0 0x000C0000
#define DYLD_IOS_VERSION_12_2 0x000C0200
#define DYLD_IOS_VERSION_13_0 0x000D0000
#define DYLD_IOS_VERSION_13_2 0x000D0200
#define DYLD_IOS_VERSION_13_4 0x000D0400
#define DYLD_IOS_VERSION_14_0 0x000E0000
#define DYLD_IOS_VERSION_14_2 0x000E0200
#define DYLD_IOS_VERSION_14_5 0x000E0500
#define DYLD_IOS_VERSION_15_0 0x000f0000
#define DYLD_IOS_VERSION_15_4 0x000f0400
#define DYLD_IOS_VERSION_16_0 0x00100000
#define DYLD_IOS_VERSION_16_4 0x00100400
#define DYLD_IOS_VERSION_17_0 0x00110000
#define DYLD_IOS_VERSION_17_2 0x00110200
#define DYLD_IOS_VERSION_17_4 0x00110400
#define DYLD_IOS_VERSION_18_0 0x00120000

#define DYLD_MACOSX_VERSION_10_10 0x000A0A00
#define DYLD_MACOSX_VERSION_10_11 0x000A0B00
#define DYLD_MACOSX_VERSION_10_12 0x000A0C00
#define DYLD_MACOSX_VERSION_10_13 0x000A0D00
#define DYLD_MACOSX_VERSION_10_13_4 0x000A0D04
#define DYLD_MACOSX_VERSION_10_14 0x000A0E00
#define DYLD_MACOSX_VERSION_10_14_4 0x000A0E04
#define DYLD_MACOSX_VERSION_10_15 0x000A0F00
#define DYLD_MACOSX_VERSION_10_15_1 0x000A0F01
#define DYLD_MACOSX_VERSION_10_15_4 0x000A0F04
#define DYLD_MACOSX_VERSION_10_16 0x000A1000
#define DYLD_MACOSX_VERSION_11_3 0x000B0300
#define DYLD_MACOSX_VERSION_12_00 0x000c0000
#define DYLD_MACOSX_VERSION_12_3 0x000c0300
#define DYLD_MACOSX_VERSION_13_0 0x000d0000
#define DYLD_MACOSX_VERSION_13_3 0x000d0300
#define DYLD_MACOSX_VERSION_14_0 0x000e0000
#define DYLD_MACOSX_VERSION_14_2 0x000e0200
#define DYLD_MACOSX_VERSION_14_4 0x000e0400
#define DYLD_MACOSX_VERSION_15_0 0x000f0000

#endif

WTF_EXTERN_C_BEGIN

// Because it is not possible to forward-declare dyld_build_version_t values,
// we forward-declare placeholders for any missing definitions, and use their empty
// value to indicate that we need to fall back to antique single-platform SDK checks.

#ifndef dyld_fall_2014_os_versions
#define dyld_fall_2014_os_versions ({ (dyld_build_version_t) { 0, 0 }; })
#endif

#ifndef dyld_fall_2015_os_versions
#define dyld_fall_2015_os_versions ({ (dyld_build_version_t) { 0, 0 }; })
#endif

#ifndef dyld_fall_2016_os_versions
#define dyld_fall_2016_os_versions ({ (dyld_build_version_t) { 0, 0 }; })
#endif

#ifndef dyld_fall_2016_os_versions
#define dyld_fall_2016_os_versions ({ (dyld_build_version_t) { 0, 0 }; })
#endif

#ifndef dyld_fall_2017_os_versions
#define dyld_fall_2017_os_versions ({ (dyld_build_version_t) { 0, 0 }; })
#endif

#ifndef dyld_spring_2018_os_versions
#define dyld_spring_2018_os_versions ({ (dyld_build_version_t) { 0, 0 }; })
#endif

#ifndef dyld_fall_2018_os_versions
#define dyld_fall_2018_os_versions ({ (dyld_build_version_t) { 0, 0 }; })
#endif

#ifndef dyld_spring_2019_os_versions
#define dyld_spring_2019_os_versions ({ (dyld_build_version_t) { 0, 0 }; })
#endif

#ifndef dyld_fall_2019_os_versions
#define dyld_fall_2019_os_versions ({ (dyld_build_version_t) { 0, 0 }; })
#endif

#ifndef dyld_late_fall_2019_os_versions
#define dyld_late_fall_2019_os_versions ({ (dyld_build_version_t) { 0, 0 }; })
#endif

#ifndef dyld_spring_2020_os_versions
#define dyld_spring_2020_os_versions ({ (dyld_build_version_t) { 0, 0 }; })
#endif

#ifndef dyld_fall_2020_os_versions
#define dyld_fall_2020_os_versions ({ (dyld_build_version_t) { 0, 0 }; })
#endif

#ifndef dyld_late_fall_2020_os_versions
#define dyld_late_fall_2020_os_versions ({ (dyld_build_version_t) { 0, 0 }; })
#endif

#ifndef dyld_spring_2021_os_versions
#define dyld_spring_2021_os_versions ({ (dyld_build_version_t) { 0, 0 }; })
#endif

#ifndef dyld_fall_2021_os_versions
#define dyld_fall_2021_os_versions ({ (dyld_build_version_t) { 0, 0 }; })
#endif

#ifndef dyld_spring_2022_os_versions
#define dyld_spring_2022_os_versions ({ (dyld_build_version_t) { 0, 0 }; })
#endif

#ifndef dyld_fall_2022_os_versions
#define dyld_fall_2022_os_versions ({ (dyld_build_version_t) { 0, 0 }; })
#endif

#ifndef dyld_2022_SU_E_os_versions
#define dyld_2022_SU_E_os_versions ({ (dyld_build_version_t) { 0, 0 }; })
#endif

#ifndef dyld_fall_2023_os_versions
#define dyld_fall_2023_os_versions ({ (dyld_build_version_t) { 0, 0 }; })
#endif

#ifndef dyld_2023_SU_C_os_versions
#define dyld_2023_SU_C_os_versions ({ (dyld_build_version_t) { 0, 0 }; })
#endif

#ifndef dyld_2023_SU_E_os_versions
#define dyld_2023_SU_E_os_versions ({ (dyld_build_version_t) { 0, 0 }; })
#endif

#ifndef dyld_fall_2024_os_versions
#define dyld_fall_2024_os_versions ({ (dyld_build_version_t) { 0, 0 }; })
#endif

uint32_t dyld_get_program_sdk_version();
bool dyld_program_sdk_at_least(dyld_build_version_t);
extern const char* dyld_shared_cache_file_path(void);
extern const struct mach_header* dyld_image_header_containing_address(const void* addr);
extern const struct mach_header* _dyld_get_dlopen_image_header(void* handle);
extern bool _dyld_get_image_uuid(const struct mach_header* mh, uuid_t);
extern bool _dyld_get_shared_cache_uuid(uuid_t);

WTF_EXTERN_C_END
