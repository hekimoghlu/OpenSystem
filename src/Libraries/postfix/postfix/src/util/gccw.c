/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 14, 2022.
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
#include <setjmp.h>

jmp_buf jbuf;

 /* -Wmissing-prototypes: no previous prototype for 'test1' */
 /* -Wimplicit: return type defaults to `int' */
test1(void)
{
    /* -Wunused: unused variable `foo' */
    int     foo;

    /* -Wparentheses: suggest parentheses around && within || */
    printf("%d\n", 1 && 2 || 3 && 4);
    /* -W: statement with no effect */
    0;
    /* BROKEN in gcc 3 */
    /* -W: control reaches end of non-void function */
}


 /* -W??????: unused parameter `foo' */
void    test2(int foo)
{
    enum {
    a = 10, b = 15} moe;
    int     bar;

    /* -Wuninitialized: 'bar' might be used uninitialized in this function */
    /* -Wformat: format argument is not a pointer (arg 2) */
    printf("%s\n", bar);
    /* -Wformat: too few arguments for format */
    printf("%s%s\n", "bar");
    /* -Wformat: too many arguments for format */
    printf("%s\n", "bar", "bar");

    /* -Wswitch: enumeration value `b' not handled in switch */
    switch (moe) {
    case a:
	return;
    }
}

 /* -Wstrict-prototypes: function declaration isn't a prototype */
void    test3()
{
}
