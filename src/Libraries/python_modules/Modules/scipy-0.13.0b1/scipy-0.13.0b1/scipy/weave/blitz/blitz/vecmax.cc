/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 30, 2024.
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
#ifndef BZ_VECMAX_CC
#define BZ_VECMAX_CC

#ifndef BZ_VECGLOBS_H
 #error <blitz/vecmax.cc> must be included via <blitz/vecglobs.h>
#endif

BZ_NAMESPACE(blitz)

template<typename P_expr>
inline
Extremum<_bz_typename P_expr::T_numtype, int> _bz_vec_max(P_expr vector)
{
    typedef _bz_typename P_expr::T_numtype T_numtype;

    T_numtype maxValue = vector(0);
    int maxIndex = 0;
    int length = vector._bz_suggestLength();

    if (vector._bz_hasFastAccess())
    {
        for (int i=1; i < length; ++i)
        {
            T_numtype value = vector._bz_fastAccess(i);
            if (value > maxValue)
            {
                maxValue = value;
                maxIndex = i;
            }
        }
    }
    else {
        for (int i=1; i < length; ++i)
        {
            T_numtype value = vector(i);
            if (value > maxValue)
            {
                maxValue = value;
                maxIndex = i;
            }
        }
    }

    return Extremum<T_numtype, int>(maxValue, maxIndex);
}

// max(vector)
template<typename P_numtype>
inline
Extremum<P_numtype, int> (max)(const Vector<P_numtype>& x)
{
    return _bz_vec_max(x._bz_asVecExpr());
}

// max(expr)
template<typename P_expr>
inline
Extremum<_bz_typename P_expr::T_numtype,int> (max)(_bz_VecExpr<P_expr> x)
{
    return _bz_vec_max(x);
}

// max(vecpick)
template<typename P_numtype>
inline
Extremum<P_numtype, int> (max)(const VectorPick<P_numtype>& x)
{
    return _bz_vec_max(x._bz_asVecExpr());
}

// max(TinyVector)
template<typename P_numtype, int N_length>
inline
Extremum<P_numtype, int> (max)(const TinyVector<P_numtype, N_length>& x)
{
    return _bz_vec_max(x._bz_asVecExpr());
}


// maxIndex(vector)
template<typename P_numtype>
inline
int maxIndex(const Vector<P_numtype>& x)
{
    return _bz_vec_max(x).index();
}

// maxIndex(expr)
template<typename P_expr>
inline
int maxIndex(_bz_VecExpr<P_expr> x)
{
    return _bz_vec_max(x._bz_asVecExpr()).index();
}

// maxIndex(vecpick)
template<typename P_numtype>
int maxIndex(const VectorPick<P_numtype>& x)
{
    return _bz_vec_max(x._bz_asVecExpr()).index();
}

// maxIndex(TinyVector)
template<typename P_numtype, int N_length>
int maxIndex(const TinyVector<P_numtype, N_length>& x)
{
    return _bz_vec_max(x._bz_asVecExpr()).index();
}

// maxValue(vector)
template<typename P_numtype>
inline
int maxValue(const Vector<P_numtype>& x)
{
    return _bz_vec_max(x._bz_asVecExpr()).value();
}

// maxValue(expr)
template<typename P_expr>
inline
int maxValue(_bz_VecExpr<P_expr> x)
{
    return _bz_vec_max(x).value();
}

// maxValue(vecpick)
template<typename P_numtype>
int maxValue(const VectorPick<P_numtype>& x)
{
    return _bz_vec_max(x._bz_asVecExpr()).value();
}

// maxValue(TinyVector)
template<typename P_numtype, int N_length>
int maxValue(const TinyVector<P_numtype, N_length>& x)
{
    return _bz_vec_max(x._bz_asVecExpr()).value();
}

BZ_NAMESPACE_END

#endif // BZ_VECMAX_CC

