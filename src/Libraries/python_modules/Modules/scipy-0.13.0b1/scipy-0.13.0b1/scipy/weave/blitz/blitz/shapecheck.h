/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 1, 2024.
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
#ifndef BZ_SHAPECHECK_H
#define BZ_SHAPECHECK_H

BZ_NAMESPACE(blitz)

/*
 * The function areShapesConformable(A,B) checks that the shapes 
 * A and B are conformable (i.e. the same size/geometry).  Typically 
 * the A and B parameters are of type TinyVector<int,N_rank> and represent 
 * the extent of the arrays.  It's possible that in the future jagged-edged
 * arrays will be supported, in which case shapes may be lists
 * of subdomains.
 */

template<typename T_shape1, typename T_shape2>
inline bool areShapesConformable(const T_shape1&, const T_shape2&)
{
    // If the shape objects are different types, this means
    // that the arrays are different ranks, or one is jagged
    // edged, etc.  In this case the two arrays are not
    // conformable.
    return false;
}

template<typename T_shape>
inline bool areShapesConformable(const T_shape& a, const T_shape& b)
{
    // The shape objects are the same type, so compare them.

    // NEEDS_WORK-- once the "all" reduction is implemented, should
    // use it.
    // return all(a == b);

    for (unsigned i=0; i < a.length(); ++i)
    {
        if (a[i] != b[i])
        {
            BZ_DEBUG_MESSAGE("Incompatible shapes detected: " << endl 
                 << a << endl << b << endl);
            return false;
        }
    }

    return true;
}

BZ_NAMESPACE_END

#endif
