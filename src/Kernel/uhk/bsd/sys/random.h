/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 29, 2023.
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
#ifndef __SYS_RANDOM_H__
#define __SYS_RANDOM_H__

#ifndef KERNEL
#include <Availability.h>
#include <stddef.h>
#endif
#include <sys/appleapiopts.h>
#include <sys/cdefs.h>

#ifndef KERNEL
__BEGIN_DECLS
int getentropy(void* buffer, size_t size) __OSX_AVAILABLE(10.12) __IOS_AVAILABLE(10.0) __TVOS_AVAILABLE(10.0) __WATCHOS_AVAILABLE(3.0);
__END_DECLS

#else /* KERNEL */
#ifdef __APPLE_API_UNSTABLE
__BEGIN_DECLS
void read_random(void* buffer, u_int numBytes);
void read_frandom(void* buffer, u_int numBytes);
int  write_random(void* buffer, u_int numBytes);
__END_DECLS
#endif /* __APPLE_API_UNSTABLE */

#endif /* KERNEL */
#endif /* __SYS_RANDOM_H__ */
