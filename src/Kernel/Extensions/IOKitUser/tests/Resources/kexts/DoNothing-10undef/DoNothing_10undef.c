/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 24, 2022.
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
#include <mach/mach_types.h>

#define UNUSED __attribute__((unused))

// These will turn into undefined symbols in the kext.  The count
// as well as the names themselves are checked in OSKext-tests.c

extern void do_nothing_1(void);
extern void do_nothing_2(void);
extern void do_nothing_3(void);
extern void do_nothing_4(void);
extern void do_nothing_5(void);
extern void do_nothing_6(void);
extern void do_nothing_7(void);
extern void do_nothing_8(void);
extern void do_nothing_9(void);
extern void do_nothing_10(void);

kern_return_t DoNothing_10undef_start(kmod_info_t * ki, void *d);
kern_return_t DoNothing_10undef_stop(kmod_info_t *ki, void *d);

kern_return_t DoNothing_10undef_start(UNUSED kmod_info_t * ki, UNUSED void *d)
{
    // reference functions to keep symbols from getting stripped.
    do_nothing_1();
    do_nothing_2();
    do_nothing_3();
    do_nothing_4();
    do_nothing_5();
    do_nothing_6();
    do_nothing_7();
    do_nothing_8();
    do_nothing_9();
    do_nothing_10();

    return KERN_SUCCESS;
}

kern_return_t DoNothing_10undef_stop(UNUSED kmod_info_t *ki, UNUSED void *d)
{
    return KERN_SUCCESS;
}
