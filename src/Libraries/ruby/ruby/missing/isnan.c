/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 4, 2022.
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
#include "ruby/missing.h"

/*
 * isnan() may be a macro, a function or both.
 * (The C99 standard defines that isnan() is a macro, though.)
 * http://www.gnu.org/software/automake/manual/autoconf/Function-Portability.html
 *
 * macro only: uClibc
 * both: GNU libc
 *
 * This file is compile if no isnan() function is available.
 * (autoconf AC_REPLACE_FUNCS detects only the function.)
 * The macro is detected by following #ifndef.
 */

#ifndef isnan
static int double_ne(double n1, double n2);

int
isnan(double n)
{
    return double_ne(n, n);
}

static int
double_ne(double n1, double n2)
{
    return n1 != n2;
}
#endif
