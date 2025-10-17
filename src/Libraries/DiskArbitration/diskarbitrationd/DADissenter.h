/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 23, 2025.
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
#ifndef __DISKARBITRATIOND_DADISSENTER__
#define __DISKARBITRATIOND_DADISSENTER__

#include <CoreFoundation/CoreFoundation.h>
#include <DiskArbitration/DiskArbitration.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

typedef const struct __DADissenter * DADissenterRef;

extern DADissenterRef DADissenterCreate( CFAllocatorRef allocator, DAReturn status );

extern pid_t DADissenterGetProcessID( DADissenterRef dissenter );

extern DAReturn DADissenterGetStatus( DADissenterRef dissenter );

extern void DADissenterSetProcessID( DADissenterRef dissenter, pid_t pid );

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* !__DISKARBITRATIOND_DADISSENTER__ */
