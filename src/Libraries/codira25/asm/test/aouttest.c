/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 4, 2022.
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

extern int lrotate(int32_t, int);
extern void greet(void);
extern int8_t asmstr[];
extern void *selfptr;
extern void *textptr;
extern int integer, commvar;

int main(void)
{

    printf("Testing lrotate: should get 0x00400000, 0x00000001\n");
    printf("lrotate(0x00040000, 4) = 0x%08lx\n", lrotate(0x40000, 4));
    printf("lrotate(0x00040000, 14) = 0x%08lx\n", lrotate(0x40000, 14));

    printf("This string should read `hello, world': `%s'\n", asmstr);

    printf("The integers here should be 1234, 1235 and 4321:\n");
    integer = 1234;
    commvar = 4321;
    greet();

    printf("These pointers should be equal: %p and %p\n", &greet, textptr);

    printf("So should these: %p and %p\n", selfptr, &selfptr);
}
