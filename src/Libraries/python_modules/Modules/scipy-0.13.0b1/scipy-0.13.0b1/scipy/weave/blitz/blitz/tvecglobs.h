/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 25, 2024.
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
#ifndef BZ_TVECGLOBS_H
#define BZ_TVECGLOBS_H

#ifndef BZ_META_METAPROG_H
 #include <blitz/meta/metaprog.h>
#endif

#ifndef BZ_NUMTRAIT_H
 #include <blitz/numtrait.h>
#endif

#include <blitz/tvcross.h>       // Cross products
#include <blitz/meta/dot.h>
#include <blitz/meta/product.h>
#include <blitz/meta/sum.h>

BZ_NAMESPACE(blitz)

template<typename T_numtype1, typename T_numtype2, int N_length>
inline BZ_PROMOTE(T_numtype1, T_numtype2)
dot(const TinyVector<T_numtype1, N_length>& a, 
    const TinyVector<T_numtype2, N_length>& b)
{
    return _bz_meta_vectorDot<N_length, 0>::f(a,b);
}

template<typename T_expr1, typename T_numtype2, int N_length>
inline BZ_PROMOTE(_bz_typename T_expr1::T_numtype, T_numtype2)
dot(_bz_VecExpr<T_expr1> a, const TinyVector<T_numtype2, N_length>& b)
{
    return _bz_meta_vectorDot<N_length, 0>::f_value_ref(a,b);
}

template<typename T_numtype1, typename T_expr2, int N_length>
inline BZ_PROMOTE(T_numtype1, _bz_typename T_expr2::T_numtype)
dot(const TinyVector<T_numtype1, N_length>& a, _bz_VecExpr<T_expr2> b)
{
    return _bz_meta_vectorDot<N_length, 0>::f_ref_value(a,b);
}

template<typename T_numtype1, int N_length>
inline BZ_SUMTYPE(T_numtype1)
product(const TinyVector<T_numtype1, N_length>& a)
{
    return _bz_meta_vectorProduct<N_length, 0>::f(a);
}

template<typename T_numtype, int N_length>
inline T_numtype
sum(const TinyVector<T_numtype, N_length>& a)
{
    return _bz_meta_vectorSum<N_length, 0>::f(a);
}

BZ_NAMESPACE_END

#endif // BZ_TVECGLOBS_H

