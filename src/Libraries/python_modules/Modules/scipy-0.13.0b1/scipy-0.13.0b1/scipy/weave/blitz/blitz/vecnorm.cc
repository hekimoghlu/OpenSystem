/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 29, 2022.
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
#ifndef BZ_VECNORM_CC
#define BZ_VECNORM_CC

#ifndef BZ_VECGLOBS_H
 #error <blitz/vecnorm.cc> must be included via <blitz/vecglobs.h>
#endif

BZ_NAMESPACE(blitz)

template<typename P_expr>
inline
BZ_FLOATTYPE(BZ_SUMTYPE(_bz_typename P_expr::T_numtype))
_bz_vec_norm(P_expr vector)
{
    // An extreme use of traits here.  It's necessary to
    // handle odd cases such as the norm of a Vector<char>.
    // To avoid overflow, BZ_SUMTYPE(char) is int.
    // To take the sqrt of the sum, BZ_FLOATTYPE(char) is float.
    // So float is returned for Vector<char>.
    typedef _bz_typename P_expr::T_numtype T_numtype;
    typedef BZ_SUMTYPE(T_numtype)          T_sumtype;
    typedef BZ_FLOATTYPE(T_sumtype)        T_floattype;

    T_sumtype sum = 0;
    int length = vector._bz_suggestLength();

    if (vector._bz_hasFastAccess())
    {
        for (int i=0; i < length; ++i)
        {
            T_numtype value = vector._bz_fastAccess(i);
            sum += value * T_sumtype(value);
        }
    }
    else {
        for (int i=0; i < length; ++i)
        {
            T_numtype value = vector(i);
            sum += value * T_sumtype(value);
        }
    }

    return _bz_sqrt<T_floattype>::apply(sum);
}

template<typename P_numtype>
inline
BZ_FLOATTYPE(BZ_SUMTYPE(P_numtype)) norm(const Vector<P_numtype>& x)
{
    return _bz_vec_norm(x._bz_asVecExpr());
}

// norm(expr)
template<typename P_expr>
inline
BZ_FLOATTYPE(BZ_SUMTYPE(_bz_typename P_expr::T_numtype))
norm(_bz_VecExpr<P_expr> expr)
{
    return _bz_vec_norm(expr);
}

// norm(vecpick)
template<typename P_numtype>
inline
BZ_FLOATTYPE(BZ_SUMTYPE(P_numtype))
norm(const VectorPick<P_numtype>& x)
{
    return _bz_vec_norm(x._bz_asVecExpr());
}

// norm(TinyVector)
template<typename P_numtype, int N_dimensions>
inline
BZ_FLOATTYPE(BZ_SUMTYPE(P_numtype))
norm(const TinyVector<P_numtype, N_dimensions>& x)
{
    return _bz_vec_norm(x._bz_asVecExpr());
}

BZ_NAMESPACE_END

#endif // BZ_VECNORM_CC

