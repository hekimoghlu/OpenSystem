/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 23, 2023.
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

/**
 * @file sys/quota.h
 * @brief The quotactl() function.
 */

#include <sys/cdefs.h>

// The uapi header uses different names from userspace, oddly.
#define if_dqblk dqblk
#define if_dqinfo dqinfo
#include <linux/quota.h>
#undef if_dqblk
#undef if_dqinfo

__BEGIN_DECLS

/**
 * [quotactl(2)](https://man7.org/linux/man-pages/man2/quotactl.2.html) manipulates disk quotas.
 *
 * Returns 0 on success, and returns -1 and sets `errno` on failure.
 *
 * Available since API level 26.
 */

#if __BIONIC_AVAILABILITY_GUARD(26)
int quotactl(int __op, const char* _Nullable __special, int __id, char* __BIONIC_COMPLICATED_NULLNESS __addr) __INTRODUCED_IN(26);
#endif /* __BIONIC_AVAILABILITY_GUARD(26) */


__END_DECLS
