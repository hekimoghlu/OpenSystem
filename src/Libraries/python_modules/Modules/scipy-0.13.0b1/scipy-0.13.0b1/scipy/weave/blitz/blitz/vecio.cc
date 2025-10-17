/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 4, 2025.
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
#ifndef BZ_VECIO_CC
#define BZ_VECIO_CC

#ifndef BZ_VECTOR_H
 #include <blitz/vector.h>
#endif

BZ_NAMESPACE(blitz)

// This version of operator<< is provided as a temporary measure
// only.  It will be revised in a future release.
// NEEDS_WORK

template<typename P_numtype>
ostream& operator<<(ostream& os, const Vector<P_numtype>& x)
{
    os << "[ ";
    for (int i=0; i < x.length(); ++i)
    {
        os << setw(10) << x[i];
        if (!((i+1)%7))
            os << endl << "  ";
    }
    os << " ]";
    return os;
}

template<typename P_expr>
ostream& operator<<(ostream& os, _bz_VecExpr<P_expr> expr)
{
    Vector<_bz_typename P_expr::T_numtype> result(expr);
    os << result;
    return os;
}

BZ_NAMESPACE_END

#endif // BZ_VECIO_CC
