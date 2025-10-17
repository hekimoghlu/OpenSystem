/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 5, 2024.
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
#ifndef BZ_VECALL_CC
#define BZ_VECALL_CC

#ifndef BZ_VECGLOBS_H
 #error <blitz/vecall.cc> must be included via <blitz/vecglobs.h>
#endif

BZ_NAMESPACE(blitz)

template<typename P_expr>
inline bool _bz_vec_all(P_expr vector)
{
    int length = vector._bz_suggestLength();

    if (vector._bz_hasFastAccess())
    {
        for (int i=0; i < length; ++i)
            if (!vector._bz_fastAccess(i))
                return false;
    }
    else {
        for (int i=0; i < length; ++i)
            if (!vector[i])
                return false;
    }

    return true;
}

template<typename P_numtype>
inline bool all(const Vector<P_numtype>& x)
{
    return _bz_vec_all(x._bz_asVecExpr());
}

template<typename P_expr>
inline bool all(_bz_VecExpr<P_expr> expr)
{
    return _bz_vec_all(expr);
}

template<typename P_numtype>
inline bool all(const VectorPick<P_numtype>& x)
{
    return _bz_vec_all(x._bz_asVecExpr());
}

template<typename P_numtype, int N_dimensions>
inline bool all(const TinyVector<P_numtype, N_dimensions>& x)
{
    return _bz_vec_all(x._bz_asVecExpr());
}

BZ_NAMESPACE_END

#endif // BZ_VECALL_CC

