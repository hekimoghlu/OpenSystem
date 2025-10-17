/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 11, 2024.
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

#include <sys/reason.h>

// FIXME: Remove this ifndef once rdar://75717715 is available on bots.
#ifndef OS_REASON_WEBKIT
#define OS_REASON_WEBKIT 31
#endif

#else

#ifdef __cplusplus
extern "C" {
#endif

int terminate_with_reason(int pid, uint32_t reasonNamespace, uint64_t reasonCode, const char *reasonString, uint64_t reasonFlags);

#ifdef __cplusplus
}
#endif

#define OS_REASON_FLAG_NO_CRASH_REPORT 0x1

#define OS_REASON_WEBKIT 31

#endif
