/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 7, 2024.
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
#include <stdio.h>
#include <inttypes.h>

int8_t text[] = "hello, world\n";

extern void function(int8_t *);
extern int bsssym, commvar;
extern void *selfptr;
extern void *selfptr2;

int main(void)
{
    printf("these should be identical: %p, %p\n",
           (int32_t)selfptr, (int32_t)&selfptr);
    printf("these should be equivalent but different: %p, %p\n",
           (int32_t)selfptr2, (int32_t)&selfptr2);
    printf("you should see \"hello, world\" twice:\n");
    bsssym = 0xF00D;
    commvar = 0xD00F;
    function(text);
    printf("this should be 0xF00E: 0x%X\n", bsssym);
    printf("this should be 0xD00E: 0x%X\n", commvar);
    return 0;
}
