/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 3, 2024.
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
#include <TargetConditionals.h>

#if !TARGET_OS_IPHONE && !TARGET_OS_DRIVERKIT

#define WEAK_SYMBOL_LD_CMD(sym, version) \
        __asm__(".section __TEXT,__const\n\t" \
                ".globl $ld$weak$os" #version "$_" #sym "\n\t" \
                "$ld$weak$os" #version "$_" #sym ":\n\t" \
                "    .byte 0\n\t" \
                ".previous")

#define ADDED_IN_10_12(sym) WEAK_SYMBOL_LD_CMD(sym, 10.11)
#define ADDED_IN_10_13(sym) WEAK_SYMBOL_LD_CMD(sym, 10.12)

ADDED_IN_10_12(getentropy);

ADDED_IN_10_12(clock_getres);
ADDED_IN_10_12(clock_gettime);
ADDED_IN_10_12(clock_settime);

ADDED_IN_10_12(basename_r);
ADDED_IN_10_12(dirname_r);

ADDED_IN_10_12(mkostemp);
ADDED_IN_10_12(mkostemps);

ADDED_IN_10_12(timingsafe_bcmp);

ADDED_IN_10_13(utimensat);

#endif /* TARGET_OS_IPHONE */
