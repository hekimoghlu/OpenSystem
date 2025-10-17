/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 6, 2024.
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
#ifndef BZ_VECNORM1_CC
#define BZ_VECNORM1_CC

#ifndef BZ_VECGLOBS_H
 #error <blitz/vecnorm1.cc> must be included via <blitz/vecglobs.h>
#endif

#include <blitz/applics.h>

BZ_NAMESPACE(blitz)

template<typename P_expr>
inline
BZ_SUMTYPE(_bz_typename P_expr::T_numtype)
_bz_vec_norm1(P_expr vector)
{
    typedef _bz_typename P_expr::T_numtype T_numtype;
    typedef BZ_SUMTYPE(T_numtype)          T_sumtype;

    T_sumtype sum = 0;
    int length = vector._bz_suggestLength();

    if (vector._bz_hasFastAccess())
    {
        for (int i=0; i < length; ++i)
            sum += _bz_abs<T_numtype>::apply(vector._bz_fastAccess(i));
    }
    else {
        for (int i=0; i < length; ++i)
            sum += _bz_abs<T_numtype>::apply(vector(i));
    }

    return sum;
}

// norm1(vector)
template<typename P_numtype>
BZ_SUMTYPE(P_numtype) norm1(const Vector<P_numtype>& x)
{
    return _bz_vec_norm1(x._bz_asVecExpr());
}

// norm1(expr)
template<typename P_expr>
BZ_SUMTYPE(_bz_typename P_expr::T_numtype) norm1(_bz_VecExpr<P_expr> expr)
{
    return _bz_vec_norm1(expr);
}

// norm1(vecpick)
template<typename P_numtype>
BZ_SUMTYPE(P_numtype) norm1(const VectorPick<P_numtype>& x)
{
    return _bz_vec_norm1(x._bz_asVecExpr());
}

// norm1(TinyVector)
template<typename P_numtype, int N_dimensions>
BZ_SUMTYPE(P_numtype) norm1(const TinyVector<P_numtype, N_dimensions>& x)
{
    return _bz_vec_norm1(x._bz_asVecExpr());
}


BZ_NAMESPACE_END

#endif // BZ_VECNORM1_CC

