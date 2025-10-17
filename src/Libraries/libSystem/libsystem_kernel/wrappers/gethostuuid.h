/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 8, 2023.
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
#ifndef __GETHOSTUUID_H
#define __GETHOSTUUID_H

#include <sys/_types/_timespec.h>
#include <sys/_types/_uuid_t.h>
#include <Availability.h>

#if defined(__IPHONE_OS_VERSION_MIN_REQUIRED) && (__IPHONE_OS_VERSION_MIN_REQUIRED < __IPHONE_7_0)
int gethostuuid(uuid_t, const struct timespec *) __OSX_AVAILABLE_BUT_DEPRECATED_MSG(__MAC_NA, __MAC_NA, __IPHONE_2_0, __IPHONE_5_0, "gethostuuid() is no longer supported");
#else
int gethostuuid(uuid_t, const struct timespec *) __OSX_AVAILABLE_STARTING(__MAC_10_5, __IPHONE_NA);
#endif

#endif /* __GETHOSTUUID_H */
