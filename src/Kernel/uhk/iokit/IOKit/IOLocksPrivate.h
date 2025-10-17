/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 18, 2023.
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
/*
 *
 */

#ifndef __IOKIT_IOLOCKS_PRIVATE_H
#define __IOKIT_IOLOCKS_PRIVATE_H

#ifndef KERNEL
#error IOLocksPrivate.h is for kernel use only
#endif

#include <sys/appleapiopts.h>

#include <IOKit/system.h>

#include <IOKit/IOReturn.h>
#include <IOKit/IOTypes.h>
#include <IOKit/IOLocks.h>

#ifdef __cplusplus
extern "C" {
#endif

IORecursiveLock *
IORecursiveLockAllocWithLockGroup( lck_grp_t * lockGroup );


#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* !__IOKIT_IOLOCKS_PRIVATE_H */
