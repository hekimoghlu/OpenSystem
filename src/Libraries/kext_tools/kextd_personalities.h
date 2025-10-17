/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 15, 2024.
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
#ifndef __KEXTD_PERSONALITIES__
#define __KEXTD_PERSONALITIES__

#include "kext_tools_util.h"

/* This function must be given the array of all kexts opened by
 * kextd from the system extensions folders. kextd tries to use
 * cache files for the system extension folders' personalities,
 * but if it can't use them all, it sends the personalities from
 * all kexts opened.
 */
OSReturn sendSystemKextPersonalitiesToKernel(CFArrayRef kexts, Boolean resetFlag);

#endif /* __KEXTD_PERSONALITIES__ */
