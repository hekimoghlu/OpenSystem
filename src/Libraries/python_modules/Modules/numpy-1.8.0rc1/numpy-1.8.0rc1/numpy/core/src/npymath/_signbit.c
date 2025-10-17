/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 1, 2024.
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
int
_npy_signbit_d(double x)
{
    union
    {
        double d;
        short s[4];
        int i[2];
    } u;

    u.d = x;

#if NPY_SIZEOF_INT == 4

#ifdef WORDS_BIGENDIAN /* defined in pyconfig.h */
    return u.i[0] < 0;
#else
    return u.i[1] < 0;
#endif

#else  /* NPY_SIZEOF_INT != 4 */

#ifdef WORDS_BIGENDIAN
    return u.s[0] < 0;
#else
    return u.s[3] < 0;
#endif

#endif  /* NPY_SIZEOF_INT */
}
