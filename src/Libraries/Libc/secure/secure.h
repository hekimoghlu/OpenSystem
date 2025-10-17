/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 11, 2024.
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
#ifndef _SECURE_H_

#include <sys/types.h>

extern void __chk_fail_overflow (void) __attribute__((__noreturn__));
extern void __chk_fail_overlap (void) __attribute__((__noreturn__));

/* Assert if a -> a+an and b -> b+bn overlap.
 * 0-lengths don't overlap anything.
 */
extern void __chk_overlap (const void *a, size_t an, const void *b, size_t bn);

/* Do we avoid the overlap check for older APIs? */
extern uint32_t __chk_assert_no_overlap;

#endif
