/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 17, 2024.
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
#ifndef BZ_VECDELTA_CC
#define BZ_VECDELTA_CC

#ifndef BZ_VECGLOBS_H
 #error <blitz/vecdelta.cc> must be included via <blitz/vecglobs.h>
#endif

BZ_NAMESPACE(blitz)

template<typename P>
inline
Vector<BZ_DIFFTYPE(_bz_typename P::T_numtype)> _bz_vec_delta(P expr)
{
    typedef _bz_typename P::T_numtype   T_numtype;
    typedef BZ_DIFFTYPE(T_numtype)      T_difftype;

    int length = expr._bz_suggestLength();
    Vector<T_difftype> z(length);
    T_numtype currentElement = 0;
    T_numtype previousElement = 0;

    if (expr._bz_hasFastAccess())
    {
        for (int i=0; i < length; ++i)
        {
            currentElement = expr._bz_fastAccess(i);
            z[i] = currentElement - previousElement;
            previousElement = currentElement;
        }
    }
    else {
        for (int i=1; i < length; ++i)
        {
            currentElement = expr(i);
            z[i] = currentElement - previousElement;
            previousElement = currentElement;
        }
    }

    return z;
}

template<typename P_numtype>
Vector<BZ_DIFFTYPE(P_numtype)> delta(const Vector<P_numtype>& x)
{
    return _bz_vec_delta(x);
}

// delta(expr)
template<typename P_expr>
Vector<BZ_DIFFTYPE(_bz_typename P_expr::T_numtype)> delta(_bz_VecExpr<P_expr> x)
{
    return _bz_vec_delta(x);
}

// delta(vecpick)
template<typename P_numtype>
Vector<BZ_DIFFTYPE(P_numtype)> delta(const VectorPick<P_numtype>& x)
{
    return _bz_vec_delta(x);
}

BZ_NAMESPACE_END

#endif // BZ_VECDELTA_CC

