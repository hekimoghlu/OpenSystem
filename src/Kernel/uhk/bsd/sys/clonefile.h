/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 22, 2023.
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
#ifndef _SYS_CLONEFILE_H_
#define _SYS_CLONEFILE_H_

/* Options for clonefile calls */
#define CLONE_NOFOLLOW      0x0001     /* Don't follow symbolic links */
#define CLONE_NOOWNERCOPY   0x0002     /* Don't copy ownership information from source */
#define CLONE_ACL           0x0004     /* Copy access control lists from source */
#define CLONE_NOFOLLOW_ANY  0x0008     /* Don't follow any symbolic links in the path */

#ifndef KERNEL

#include <sys/cdefs.h>
#include <machine/_types.h>
#include <_types/_uint32_t.h>
#include <Availability.h>

__BEGIN_DECLS

int clonefileat(int, const char *, int, const char *, uint32_t) __OSX_AVAILABLE(10.12) __IOS_AVAILABLE(10.0) __TVOS_AVAILABLE(10.0) __WATCHOS_AVAILABLE(3.0);

int fclonefileat(int, int, const char *, uint32_t) __OSX_AVAILABLE(10.12) __IOS_AVAILABLE(10.0) __TVOS_AVAILABLE(10.0) __WATCHOS_AVAILABLE(3.0);

int clonefile(const char *, const char *, uint32_t) __OSX_AVAILABLE(10.12) __IOS_AVAILABLE(10.0) __TVOS_AVAILABLE(10.0) __WATCHOS_AVAILABLE(3.0);

__END_DECLS

#endif /* KERNEL */

#endif /* _SYS_CLONEFILE_H_ */
