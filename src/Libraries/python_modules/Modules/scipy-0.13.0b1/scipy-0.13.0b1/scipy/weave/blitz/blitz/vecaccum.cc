/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 1, 2022.
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
#ifndef BZ_VECACCUM_CC
#define BZ_VECACCUM_CC

#ifndef BZ_VECGLOBS_H
 #error <blitz/vecaccum.cc> must be included via <blitz/vecglobs.h>
#endif

BZ_NAMESPACE(blitz)

template<typename P>
inline
Vector<BZ_SUMTYPE(_bz_typename P::T_numtype)> _bz_vec_accumulate(P expr)
{
    typedef BZ_SUMTYPE(_bz_typename P::T_numtype) T_sumtype;
    int length = expr._bz_suggestLength();
    Vector<T_sumtype> z(length);
    T_sumtype sum = 0;

    if (expr._bz_hasFastAccess())
    {
        for (int i=0; i < length; ++i)
        {
            sum += expr._bz_fastAccess(i);
            z[i] = sum;
        }
    }
    else {
        for (int i=0; i < length; ++i)
        {
            sum += expr(i);
            z[i] = sum;
        }
    }

    return z;
}
template<typename P_numtype>
Vector<BZ_SUMTYPE(P_numtype)> accumulate(const Vector<P_numtype>& x)
{
    return _bz_vec_accumulate(x);
}

template<typename P_expr>
Vector<BZ_SUMTYPE(_bz_typename P_expr::T_numtype)>
accumulate(_bz_VecExpr<P_expr> x)
{
    return _bz_vec_accumulate(x);
}

template<typename P_numtype>
Vector<BZ_SUMTYPE(P_numtype)> accumulate(const VectorPick<P_numtype>& x)
{
    return _bz_vec_accumulate(x);
}

BZ_NAMESPACE_END

#endif // BZ_VECACCUM_CC

