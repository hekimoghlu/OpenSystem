/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 1, 2023.
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

//
// Copyright 2021 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// apple_platform.h: Apple operating system specific includes and defines.
//

#ifndef COMMON_APPLE_PLATFORM_H_
#define COMMON_APPLE_PLATFORM_H_

#import "common/platform.h"

#if ((ANGLE_PLATFORM_MACOS && __MAC_OS_X_VERSION_MIN_REQUIRED >= 120000) ||   \
     (((ANGLE_PLATFORM_IOS_FAMILY && !ANGLE_PLATFORM_IOS_FAMILY_SIMULATOR) || \
       ANGLE_PLATFORM_MACCATALYST) &&                                         \
      __IPHONE_OS_VERSION_MIN_REQUIRED >= 150000) ||                          \
     (ANGLE_PLATFORM_WATCHOS && !ANGLE_PLATFORM_IOS_FAMILY_SIMULATOR &&       \
      __WATCH_OS_VERSION_MIN_REQUIRED >= 80000) ||                            \
     (TARGET_OS_TV && !ANGLE_PLATFORM_IOS_FAMILY_SIMULATOR &&                 \
      __TV_OS_VERSION_MIN_REQUIRED >= 150000)) &&                             \
    (defined(__has_include) && __has_include(<Metal/MTLResource_Private.h>))
#    define ANGLE_HAVE_MTLRESOURCE_SET_OWNERSHIP_IDENTITY 1
#else
#    define ANGLE_HAVE_MTLRESOURCE_SET_OWNERSHIP_IDENTITY 0
#endif

#if (ANGLE_HAVE_MTLRESOURCE_SET_OWNERSHIP_IDENTITY && \
     defined(ANGLE_ENABLE_METAL_OWNERSHIP_IDENTITY))
#    define ANGLE_USE_METAL_OWNERSHIP_IDENTITY 1
#else
#    define ANGLE_USE_METAL_OWNERSHIP_IDENTITY 0
#endif

#endif /* COMMON_APPLE_PLATFORM_H_ */
