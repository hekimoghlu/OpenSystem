/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 28, 2025.
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
void
rb_sparc_flush_register_windows(void)
{
/*
 * gcc doesn't provide "asm" keyword if -ansi and the various -std options
 * are given.
 * http://gcc.gnu.org/onlinedocs/gcc/Alternate-Keywords.html
 */
#ifndef __GNUC__
#define __asm__ asm
#endif

    __asm__
#ifdef __GNUC__
    __volatile__
#endif

/* This condition should be in sync with one in configure.ac */
#if defined(__sparcv9) || defined(__sparc_v9__) || defined(__arch64__)
# ifdef __GNUC__
    ("flushw" : : : "%o7")
# else
    ("flushw")
# endif /* __GNUC__ */
#else
    ("ta 0x03")
#endif
    ;
}
