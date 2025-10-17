/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 23, 2024.
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
#ifndef BZ_MINMAX_H
#define BZ_MINMAX_H

#include <blitz/promote.h>

BZ_NAMESPACE(blitz)

/*
 * These functions are in their own namespace (blitz::minmax) to avoid
 * conflicts with the array reduction operations min and max.
 */

BZ_NAMESPACE(minmax)

template<typename T1, typename T2>
BZ_PROMOTE(T1,T2) min(const T1& a, const T2& b)
{
    typedef BZ_PROMOTE(T1,T2) T_promote;

    if (a <= b)
        return T_promote(a);
    else
        return T_promote(b);
}

template<typename T1, typename T2>
BZ_PROMOTE(T1,T2) max(const T1& a, const T2& b)
{
    typedef BZ_PROMOTE(T1,T2) T_promote;

    if (a >= b)
        return T_promote(a);
    else
        return T_promote(b);
}

BZ_NAMESPACE_END

BZ_NAMESPACE_END

#endif
