/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 15, 2023.
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
#ifndef BZ_ARRAYMISC_CC
#define BZ_ARRAYMISC_CC

#ifndef BZ_ARRAY_H
 #error <blitz/array/misc.cc> must be included via <blitz/array.h>
#endif

BZ_NAMESPACE(blitz)

#define BZ_ARRAY_DECLARE_UOP(fn, fnobj)                                \
template<typename T_numtype, int N_rank>                                  \
inline                                                                 \
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<FastArrayIterator<T_numtype,N_rank>, \
    fnobj<T_numtype> > >                                               \
fn(const Array<T_numtype,N_rank>& array)                               \
{                                                                      \
    return _bz_ArrayExprUnaryOp<FastArrayIterator<T_numtype,N_rank>,   \
        fnobj<T_numtype> >(array.beginFast());                         \
}                                                                      \
                                                                       \
template<typename T_expr>                                                 \
inline                                                                 \
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<_bz_ArrayExpr<T_expr>,              \
    fnobj<_bz_typename T_expr::T_numtype> > >                          \
fn(BZ_ETPARM(_bz_ArrayExpr<T_expr>) expr)                              \
{                                                                      \
    return _bz_ArrayExprUnaryOp<_bz_ArrayExpr<T_expr>,                 \
        fnobj<_bz_typename T_expr::T_numtype> >(expr);                 \
}                                                                      
                                                                       
BZ_ARRAY_DECLARE_UOP(operator!, LogicalNot)
BZ_ARRAY_DECLARE_UOP(operator~, BitwiseNot)
BZ_ARRAY_DECLARE_UOP(operator-, Negate)

/*
 * cast() functions, for explicit type casting
 */

template<typename T_numtype, int N_rank, typename T_cast>
inline                                                                 
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<FastArrayIterator<T_numtype,N_rank>,   
    Cast<T_numtype, T_cast> > >
cast(const Array<T_numtype,N_rank>& array, T_cast)
{                                                                      
    return _bz_ArrayExprUnaryOp<FastArrayIterator<T_numtype,N_rank>,      
        Cast<T_numtype,T_cast> >(array.beginFast());                            
}                                                                      
                                                                       
template<typename T_expr, typename T_cast>
inline                                                                 
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<_bz_ArrayExpr<T_expr>,              
    Cast<_bz_typename T_expr::T_numtype,T_cast> > >                          
cast(BZ_ETPARM(_bz_ArrayExpr<T_expr>) expr, T_cast)
{                                                                      
    return _bz_ArrayExprUnaryOp<_bz_ArrayExpr<T_expr>,                 
        Cast<_bz_typename T_expr::T_numtype,T_cast> >(expr);                 
}                                                                      
                                                                       
BZ_NAMESPACE_END

#endif // BZ_ARRAYMISC_CC

