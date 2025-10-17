/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 12, 2025.
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
#ifndef BZ_TINYVEC_CC
#define BZ_TINYVEC_CC

#include <blitz/tinyvec.h>
#include <blitz/vector.h>
#include <blitz/range.h>
#include <blitz/meta/vecassign.h>

BZ_NAMESPACE(blitz)

template<typename P_numtype, int N_length>
inline TinyVector<P_numtype, N_length>::TinyVector(const T_numtype initValue) {
    for (int i=0; i < N_length; ++i)
        data_[i] = initValue;
}

template<typename P_numtype, int N_length>
inline TinyVector<P_numtype, N_length>::TinyVector(const TinyVector<T_numtype, N_length>& x) {
    for (int i=0; i < N_length; ++i)
        data_[i] = x.data_[i];
}

template<typename P_numtype, int N_length>
template<typename P_numtype2>
inline TinyVector<P_numtype, N_length>::TinyVector(const TinyVector<P_numtype2, N_length>& x) {
    for (int i=0; i < N_length; ++i)
        data_[i] = static_cast<P_numtype>(x[i]);
}

template<typename P_numtype, int N_length>
template<typename P_expr, typename P_updater>
inline void TinyVector<P_numtype, N_length>::_bz_assign(P_expr expr, P_updater up) {
    BZPRECHECK(expr.length(N_length) == N_length,
        "An expression with length " << expr.length(N_length)
        << " was assigned to a TinyVector<"
        << BZ_DEBUG_TEMPLATE_AS_STRING_LITERAL(T_numtype)
        << "," << N_length << ">");

    if (expr._bz_hasFastAccess()) {
        _bz_meta_vecAssign<N_length, 0>::fastAssign(*this, expr, up);
    } else {
        _bz_meta_vecAssign<N_length, 0>::assign(*this, expr, up);
    }
}

// Constructor added by Peter Nordlund (peter.nordlund@ind.af.se)
template<typename P_numtype, int N_length> template<typename P_expr>
inline TinyVector<P_numtype, N_length>::TinyVector(_bz_VecExpr<P_expr> expr) 
{
  _bz_assign(expr, _bz_update<T_numtype, _bz_typename P_expr::T_numtype>());
}

/*****************************************************************************
 * Assignment operators with vector expression operand
 */

template<typename P_numtype, int N_length> template<typename P_expr>
inline TinyVector<P_numtype, N_length>& 
TinyVector<P_numtype, N_length>::operator=(_bz_VecExpr<P_expr> expr)
{
    _bz_assign(expr, _bz_update<T_numtype, _bz_typename P_expr::T_numtype>());
    return *this;
}

template<typename P_numtype, int N_length> template<typename P_expr>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator+=(_bz_VecExpr<P_expr> expr)
{
    _bz_assign(expr, _bz_plus_update<T_numtype, 
        _bz_typename P_expr::T_numtype>());
    return *this;
}

template<typename P_numtype, int N_length> template<typename P_expr>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator-=(_bz_VecExpr<P_expr> expr)
{
    _bz_assign(expr, _bz_minus_update<T_numtype,
        _bz_typename P_expr::T_numtype>());
    return *this;
}

template<typename P_numtype, int N_length> template<typename P_expr>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator*=(_bz_VecExpr<P_expr> expr)
{
    _bz_assign(expr, _bz_multiply_update<T_numtype,
        _bz_typename P_expr::T_numtype>());
    return *this;
}

template<typename P_numtype, int N_length> template<typename P_expr>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator/=(_bz_VecExpr<P_expr> expr)
{
    _bz_assign(expr, _bz_divide_update<T_numtype,
        _bz_typename P_expr::T_numtype>());
    return *this;
}

template<typename P_numtype, int N_length> template<typename P_expr>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator%=(_bz_VecExpr<P_expr> expr)
{
    _bz_assign(expr, _bz_mod_update<T_numtype,
        _bz_typename P_expr::T_numtype>());
    return *this;
}

template<typename P_numtype, int N_length> template<typename P_expr>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator^=(_bz_VecExpr<P_expr> expr)
{
    _bz_assign(expr, _bz_xor_update<T_numtype,
        _bz_typename P_expr::T_numtype>());
    return *this;
}

template<typename P_numtype, int N_length> template<typename P_expr>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator&=(_bz_VecExpr<P_expr> expr)
{
    _bz_assign(expr, _bz_bitand_update<T_numtype,
        _bz_typename P_expr::T_numtype>());
    return *this;
}

template<typename P_numtype, int N_length> template<typename P_expr>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator|=(_bz_VecExpr<P_expr> expr)
{
    _bz_assign(expr, _bz_bitor_update<T_numtype,
        _bz_typename P_expr::T_numtype>());
    return *this;
}

template<typename P_numtype, int N_length> template<typename P_expr>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator<<=(_bz_VecExpr<P_expr> expr)
{
    _bz_assign(expr, _bz_shiftl_update<T_numtype,
        _bz_typename P_expr::T_numtype>());
    return *this;
}

template<typename P_numtype, int N_length> template<typename P_expr>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator>>=(_bz_VecExpr<P_expr> expr)
{
    _bz_assign(expr, _bz_shiftr_update<T_numtype,
        _bz_typename P_expr::T_numtype>());
    return *this;
}

/*****************************************************************************
 * Assignment operators with scalar operand
 */

template<typename P_numtype, int N_length> 
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::initialize(const T_numtype x)
{
#ifndef BZ_KCC_COPY_PROPAGATION_KLUDGE
    typedef _bz_VecExprConstant<T_numtype> T_expr;
    (*this) = _bz_VecExpr<T_expr>(T_expr(x));
#else
    // Avoid using the copy propagation kludge for this simple
    // operation.
    for (int i=0; i < N_length; ++i)
        data_[i] = x;
#endif
    return *this;
}

template<typename P_numtype, int N_length>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator+=(const T_numtype x)
{
    typedef _bz_VecExprConstant<T_numtype> T_expr;
    (*this) += _bz_VecExpr<T_expr>(T_expr(x));
    return *this;
}

template<typename P_numtype, int N_length>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator-=(const T_numtype x)
{
    typedef _bz_VecExprConstant<T_numtype> T_expr;
    (*this) -= _bz_VecExpr<T_expr>(T_expr(x));
    return *this;
}

template<typename P_numtype, int N_length>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator*=(const T_numtype x)
{
    typedef _bz_VecExprConstant<T_numtype> T_expr;
    (*this) *= _bz_VecExpr<T_expr>(T_expr(x));
    return *this;
}

template<typename P_numtype, int N_length>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator/=(const T_numtype x)
{
    typedef _bz_VecExprConstant<T_numtype> T_expr;
    (*this) /= _bz_VecExpr<T_expr>(T_expr(x));
    return *this;
}

template<typename P_numtype, int N_length>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator%=(const T_numtype x)
{
    typedef _bz_VecExprConstant<T_numtype> T_expr;
    (*this) %= _bz_VecExpr<T_expr>(T_expr(x));
    return *this;
}

template<typename P_numtype, int N_length>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator^=(const T_numtype x)
{
    typedef _bz_VecExprConstant<T_numtype> T_expr;
    (*this) ^= _bz_VecExpr<T_expr>(T_expr(x));
    return *this;
}

template<typename P_numtype, int N_length>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator&=(const T_numtype x)
{
    typedef _bz_VecExprConstant<T_numtype> T_expr;
    (*this) &= _bz_VecExpr<T_expr>(T_expr(x));
    return *this;
}

template<typename P_numtype, int N_length>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator|=(const T_numtype x)
{
    typedef _bz_VecExprConstant<T_numtype> T_expr;
    (*this) |= _bz_VecExpr<T_expr>(T_expr(x));
    return *this;
}

template<typename P_numtype, int N_length>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator<<=(const int x)
{
    typedef _bz_VecExprConstant<int> T_expr;
    (*this) <<= _bz_VecExpr<T_expr>(T_expr(x));
    return *this;
}

template<typename P_numtype, int N_length>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator>>=(const int x)
{
    typedef _bz_VecExprConstant<int> T_expr;
    (*this) >>= _bz_VecExpr<T_expr>(T_expr(x));
    return *this;
}

/*****************************************************************************
 * Assignment operators with TinyVector operand
 */

template<typename P_numtype, int N_length> template<typename P_numtype2>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator=(const TinyVector<P_numtype2, N_length>& x) {
    (*this) = _bz_VecExpr<_bz_typename 
        TinyVector<P_numtype2, N_length>::T_constIterator>(x.beginFast());
    return *this;
}

template<typename P_numtype, int N_length> template<typename P_numtype2>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator+=(const TinyVector<P_numtype2, N_length>& x) {
    (*this) += _bz_VecExpr<_bz_typename
        TinyVector<P_numtype2, N_length>::T_constIterator>(x.beginFast());
    return *this;
}

template<typename P_numtype, int N_length> template<typename P_numtype2>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator-=(const TinyVector<P_numtype2, N_length>& x) {
    (*this) -= _bz_VecExpr<_bz_typename
        TinyVector<P_numtype2, N_length>::T_constIterator>(x.beginFast());
    return *this;
}

template<typename P_numtype, int N_length> template<typename P_numtype2>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator*=(const TinyVector<P_numtype2, N_length>& x) {
    (*this) *= _bz_VecExpr<_bz_typename
        TinyVector<P_numtype2, N_length>::T_constIterator>(x.beginFast());
    return *this;
}

template<typename P_numtype, int N_length> template<typename P_numtype2>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator/=(const TinyVector<P_numtype2, N_length>& x) {
    (*this) /= _bz_VecExpr<_bz_typename
        TinyVector<P_numtype2, N_length>::T_constIterator>(x.beginFast());
    return *this;
}

template<typename P_numtype, int N_length> template<typename P_numtype2>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator%=(const TinyVector<P_numtype2, N_length>& x) {
    (*this) %= _bz_VecExpr<_bz_typename
        TinyVector<P_numtype2, N_length>::T_constIterator>(x.beginFast());
    return *this;
}

template<typename P_numtype, int N_length> template<typename P_numtype2>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator^=(const TinyVector<P_numtype2, N_length>& x) {
    (*this) ^= _bz_VecExpr<_bz_typename
        TinyVector<P_numtype2, N_length>::T_constIterator>(x.beginFast());
    return *this;
}

template<typename P_numtype, int N_length> template<typename P_numtype2>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator&=(const TinyVector<P_numtype2, N_length>& x) {
    (*this) &= _bz_VecExpr<_bz_typename
        TinyVector<P_numtype2, N_length>::T_constIterator>(x.beginFast());
    return *this;
}

template<typename P_numtype, int N_length> template<typename P_numtype2>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator|=(const TinyVector<P_numtype2, N_length>& x) {
    (*this) |= _bz_VecExpr<_bz_typename
        TinyVector<P_numtype2, N_length>::T_constIterator>(x.beginFast());
    return *this;
}

template<typename P_numtype, int N_length> template<typename P_numtype2>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator<<=(const TinyVector<P_numtype2, N_length>& x) {
    (*this) <<= _bz_VecExpr<_bz_typename
        TinyVector<P_numtype2, N_length>::T_constIterator>(x.beginFast());
    return *this;
}

template<typename P_numtype, int N_length> template<typename P_numtype2>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator>>=(const TinyVector<P_numtype2, N_length>& x) {
    (*this) >>= _bz_VecExpr<_bz_typename
        TinyVector<P_numtype2, N_length>::T_constIterator>(x.beginFast());
    return *this;
}

/*****************************************************************************
 * Assignment operators with Vector operand
 */

template<typename P_numtype, int N_length> template<typename P_numtype2>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator=(const Vector<P_numtype2>& x) {
    (*this) = x._bz_asVecExpr();
    return *this;
}

template<typename P_numtype, int N_length> template<typename P_numtype2>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator+=(const Vector<P_numtype2>& x) {
    (*this) += x._bz_asVecExpr();
    return *this;
}

template<typename P_numtype, int N_length> template<typename P_numtype2>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator-=(const Vector<P_numtype2>& x) {
    (*this) -= x._bz_asVecExpr();
    return *this;
}

template<typename P_numtype, int N_length> template<typename P_numtype2>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator*=(const Vector<P_numtype2>& x) {
    (*this) *= x._bz_asVecExpr();
    return *this;
}

template<typename P_numtype, int N_length> template<typename P_numtype2>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator/=(const Vector<P_numtype2>& x) {
    (*this) /= x._bz_asVecExpr();
    return *this;
}

template<typename P_numtype, int N_length> template<typename P_numtype2>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator%=(const Vector<P_numtype2>& x) {
    (*this) %= x._bz_asVecExpr();
    return *this;
}

template<typename P_numtype, int N_length> template<typename P_numtype2>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator^=(const Vector<P_numtype2>& x) {
    (*this) ^= x._bz_asVecExpr();
    return *this;
}

template<typename P_numtype, int N_length> template<typename P_numtype2>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator&=(const Vector<P_numtype2>& x) {
    (*this) &= x._bz_asVecExpr();
    return *this;
}

template<typename P_numtype, int N_length> template<typename P_numtype2>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator|=(const Vector<P_numtype2>& x) {
    (*this) |= x._bz_asVecExpr();
    return *this;
}

template<typename P_numtype, int N_length> template<typename P_numtype2>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator<<=(const Vector<P_numtype2>& x) {
    (*this) <<= x._bz_asVecExpr();
    return *this;
}

template<typename P_numtype, int N_length> template<typename P_numtype2>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator>>=(const Vector<P_numtype2>& x) {
    (*this) >>= x._bz_asVecExpr();
    return *this;
}

/*****************************************************************************
 * Assignment operators with Range operand
 */

template<typename P_numtype, int N_length> 
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator=(const Range& r) {
    (*this) = r._bz_asVecExpr();
    return *this;
}

template<typename P_numtype, int N_length>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator+=(const Range& r) {
    (*this) += r._bz_asVecExpr();
    return *this;
}

template<typename P_numtype, int N_length>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator-=(const Range& r) {
    (*this) -= r._bz_asVecExpr();
    return *this;
}

template<typename P_numtype, int N_length>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator*=(const Range& r) {
    (*this) *= r._bz_asVecExpr();
    return *this;
}

template<typename P_numtype, int N_length>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator/=(const Range& r) {
    (*this) /= r._bz_asVecExpr();
    return *this;
}

template<typename P_numtype, int N_length>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator%=(const Range& r) {
    (*this) %= r._bz_asVecExpr();
    return *this;
}

template<typename P_numtype, int N_length>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator^=(const Range& r) {
    (*this) ^= r._bz_asVecExpr();
    return *this;
}

template<typename P_numtype, int N_length>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator&=(const Range& r) {
    (*this) &= r._bz_asVecExpr();
    return *this;
}

template<typename P_numtype, int N_length>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator|=(const Range& r) {
    (*this) |= r._bz_asVecExpr();
    return *this;
}

template<typename P_numtype, int N_length>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator<<=(const Range& r) {
    (*this) <<= r._bz_asVecExpr();
    return *this;
}

template<typename P_numtype, int N_length>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator>>=(const Range& r) {
    (*this) >>= r._bz_asVecExpr();
    return *this;
}

/*****************************************************************************
 * Assignment operators with VectorPick operand
 */

template<typename P_numtype, int N_length> template<typename P_numtype2>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator=(const VectorPick<P_numtype2>& x)
{
    (*this) = x._bz_asVecExpr();
    return *this;
}

template<typename P_numtype, int N_length> template<typename P_numtype2>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator+=(const VectorPick<P_numtype2>& x)
{
    (*this) += x._bz_asVecExpr();
    return *this;
}

template<typename P_numtype, int N_length> template<typename P_numtype2>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator-=(const VectorPick<P_numtype2>& x)
{
    (*this) -= x._bz_asVecExpr();
    return *this;
}

template<typename P_numtype, int N_length> template<typename P_numtype2>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator*=(const VectorPick<P_numtype2>& x)
{
    (*this) *= x._bz_asVecExpr();
    return *this;
}

template<typename P_numtype, int N_length> template<typename P_numtype2>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator/=(const VectorPick<P_numtype2>& x)
{
    (*this) /= x._bz_asVecExpr();
    return *this;
}

template<typename P_numtype, int N_length> template<typename P_numtype2>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator%=(const VectorPick<P_numtype2>& x)
{
    (*this) %= x._bz_asVecExpr();
    return *this;
}

template<typename P_numtype, int N_length> template<typename P_numtype2>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator^=(const VectorPick<P_numtype2>& x)
{
    (*this) ^= x._bz_asVecExpr();
    return *this;
}

template<typename P_numtype, int N_length> template<typename P_numtype2>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator&=(const VectorPick<P_numtype2>& x)
{
    (*this) &= x._bz_asVecExpr();
    return *this;
}

template<typename P_numtype, int N_length> template<typename P_numtype2>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator|=(const VectorPick<P_numtype2>& x)
{
    (*this) |= x._bz_asVecExpr();
    return *this;
}

template<typename P_numtype, int N_length> template<typename P_numtype2>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator>>=(const VectorPick<P_numtype2>& x)
{
    (*this) <<= x._bz_asVecExpr();
    return *this;
}

BZ_NAMESPACE_END

#endif // BZ_TINYVEC_CC
