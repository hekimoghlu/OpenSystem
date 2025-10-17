/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 12, 2024.
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
#ifndef _PRIVATE_BIONIC_ARC4RANDOM_H_
#define _PRIVATE_BIONIC_ARC4RANDOM_H_

#include <stddef.h>

// arc4random(3) aborts if it's unable to fetch entropy, which is always
// the case for init on devices. GCE kernels have a workaround to ensure
// sufficient entropy during early boot, but no device kernels do. This
// wrapper falls back to AT_RANDOM if the kernel doesn't have enough
// entropy for getrandom(2) or /dev/urandom.
void __libc_safe_arc4random_buf(void* buf, size_t n);

#endif
