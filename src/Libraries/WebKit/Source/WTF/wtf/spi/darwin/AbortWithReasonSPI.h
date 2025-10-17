/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 15, 2022.
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

#ifdef __cplusplus
extern "C" {
#endif

#if (PLATFORM(MAC) && __MAC_OS_X_VERSION_MIN_REQUIRED >= 130000) \
    || ((PLATFORM(IOS) || PLATFORM(MACCATALYST)) && __IPHONE_OS_VERSION_MIN_REQUIRED >= 160000) \
    || (PLATFORM(WATCHOS) && __WATCH_OS_VERSION_MIN_REQUIRED >= 90000) \
    || (PLATFORM(APPLETV) && __TV_OS_VERSION_MIN_REQUIRED >= 160000) \
    || PLATFORM(VISION)
#if USE(APPLE_INTERNAL_SDK) && !defined(__swift__)
#include <sys/reason.h>
#else
void abort_with_reason(uint32_t reason_namespace, uint64_t reason_code, const char *reason_string, uint64_t reason_flags);
#endif

#else

#define abort_with_reason(reason_namespace, reason_code, reason_string, reason_flags)  CRASH()

#endif

#ifdef __cplusplus
}
#endif

#if !USE(APPLE_INTERNAL_SDK) || defined(__swift__)
#define OS_REASON_FLAG_NO_CRASH_REPORT     0x1
#define OS_REASON_FLAG_SECURITY_SENSITIVE  0x1000
#define OS_REASON_WEBKIT 31
#endif
