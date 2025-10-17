/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 28, 2022.
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
#ifndef _SYS_STDIO_H_
#define _SYS_STDIO_H_

#include <sys/cdefs.h>

#if __DARWIN_C_LEVEL >= __DARWIN_C_FULL
#define RENAME_SECLUDE                  0x00000001
#define RENAME_SWAP                     0x00000002
#define RENAME_EXCL                     0x00000004
#define RENAME_RESERVED1                0x00000008
#define RENAME_NOFOLLOW_ANY             0x00000010
#endif

#ifndef KERNEL
#if __DARWIN_C_LEVEL >= 200809L
#include <Availability.h>

__BEGIN_DECLS

int     renameat(int, const char *, int, const char *) __OSX_AVAILABLE_STARTING(__MAC_10_10, __IPHONE_8_0);

#if __DARWIN_C_LEVEL >= __DARWIN_C_FULL

int renamex_np(const char *, const char *, unsigned int) __OSX_AVAILABLE(10.12) __IOS_AVAILABLE(10.0) __TVOS_AVAILABLE(10.0) __WATCHOS_AVAILABLE(3.0);
int renameatx_np(int, const char *, int, const char *, unsigned int) __OSX_AVAILABLE(10.12) __IOS_AVAILABLE(10.0) __TVOS_AVAILABLE(10.0) __WATCHOS_AVAILABLE(3.0);

#endif /* __DARWIN_C_LEVEL >= __DARWIN_C_FULL */

__END_DECLS

#endif /* __DARWIN_C_LEVEL >= 200809L */

#endif /* !KERNEL */
#endif /* _SYS_STDIO_H_ */
