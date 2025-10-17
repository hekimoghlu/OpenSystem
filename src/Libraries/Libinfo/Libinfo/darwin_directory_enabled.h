/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 7, 2024.
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
// Copyright (c) 2023 Apple Inc. All rights reserved.
//

#ifndef DARWIN_DIRECTORY_ENABLED_H
#define DARWIN_DIRECTORY_ENABLED_H

#include <TargetConditionals.h>
#include <os/feature_private.h>
#include <os/variant_private.h>

OS_ALWAYS_INLINE
static inline bool
_darwin_directory_enabled(void)
{
#ifdef DARWIN_DIRECTORY_AVAILABLE
#if TARGET_OS_OSX
    return os_feature_enabled_simple(DarwinDirectory, LibinfoLookups_macOS, false) &&
           os_variant_is_darwinos("com.apple.DarwinDirectory");
#else
    return os_feature_enabled_simple(DarwinDirectory, LibinfoLookups, false);
#endif // if TARGET_OS_OSX
#endif // ifdef DARWIN_DIRECTORY_AVAILABLE
    return false;
}

#endif // DARWIN_DIRECTORY_ENABLED_H
