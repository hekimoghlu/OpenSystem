/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 6, 2025.
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
#ifndef BZ_MATEXPR_H
#define BZ_MATEXPR_H

#ifndef BZ_MATRIX_H
 #error <blitz/matexpr.h> must be included via <blitz/matrix.h>
#endif

#include <blitz/applics.h>

BZ_NAMESPACE(blitz)

// BlitzMatrixExpressionsBase is a dummy class provided for users of
// graphical class browsers.  
class BlitzMatrixExpressionsBase { };

template<typename P_expr>
class _bz_MatExpr : public BlitzMatrixExpressionsBase {

public:
    typedef P_expr T_expr;
    typedef _bz_typename T_expr::T_numtype T_numtype;

#ifdef BZ_PASS_EXPR_BY_VALUE
    _bz_MatExpr(T_expr a)
        : iter_(a)
    { }
#else
    _bz_MatExpr(const T_expr& a)
        : iter_(a)
    { }
#endif

    T_numtype operator()(unsigned i, unsigned j) const
    { return iter_(i,j); }

    unsigned rows(unsigned recommendedRows) const
    { return iter_.rows(recommendedRows); }

    unsigned cols(unsigned recommendedCols) const
    { return iter_.cols(recommendedCols); }

private:
    T_expr iter_;
};

template<typename P_expr1, typename P_expr2, typename P_op>
class _bz_MatExprOp : public BlitzMatrixExpressionsBase {

public:
    typedef P_expr1 T_expr1;
    typedef P_expr2 T_expr2;
    typedef _bz_typename T_expr1::T_numtype T_numtype1;
    typedef _bz_typename T_expr2::T_numtype T_numtype2;
    typedef BZ_PROMOTE(T_numtype1, T_numtype2) T_numtype;
    typedef P_op    T_op;

#ifdef BZ_PASS_EXPR_BY_VALUE
    _bz_MatExprOp(T_expr1 a, T_expr2 b)
        : iter1_(a), iter2_(b)
    { }
#else
    _bz_MatExprOp(const T_expr1& a, const T_expr2& b)
        : iter1_(a), iter2_(b)
    { }
#endif

    T_numtype operator()(unsigned i, unsigned j) const
    { return T_op::apply(iter1_(i,j), iter2_(i,j)); }

    unsigned rows(unsigned recommendedRows) const
    {
        BZPRECONDITION(iter2_.rows(recommendedRows) == 
            iter1_.rows(recommendedRows));
        return iter1_.rows(recommendedRows);
    }

    unsigned cols(unsigned recommendedCols) const
    {
        BZPRECONDITION(iter2_.cols(recommendedCols) == 
            iter1_.cols(recommendedCols));
        return iter1_.cols(recommendedCols);
    }

private:
    T_expr1 iter1_;
    T_expr2 iter2_;
};

template<typename P_expr, typename P_unaryOp>
class _bz_MatExprUnaryOp : public BlitzMatrixExpressionsBase {

public:
    typedef P_expr T_expr;
    typedef P_unaryOp T_unaryOp;
    typedef _bz_typename T_unaryOp::T_numtype T_numtype;

#ifdef BZ_PASS_EXPR_BY_VALUE
    _bz_MatExprUnaryOp(T_expr iter)
        : iter_(iter)
    { }
#else
    _bz_MatExprUnaryOp(const T_expr& iter)
        : iter_(iter)
    { }
#endif

    T_numtype operator()(unsigned i, unsigned j) const
    { return T_unaryOp::apply(iter_(i,j)); }

    unsigned rows(unsigned recommendedRows) const
    { return iter_.rows(recommendedRows); }

    unsigned cols(unsigned recommendedCols) const
    { return iter_.cols(recommendedCols); }

private:
    T_expr iter_;    
};

template<typename P_numtype>
class _bz_MatExprConstant : public BlitzMatrixExpressionsBase {
public:
    typedef P_numtype T_numtype;

    _bz_MatExprConstant(P_numtype value)
        : value_(value)
    { }

    T_numtype operator()(unsigned i, unsigned j) const
    { return value_; }

    unsigned rows(unsigned recommendedRows) const
    { return recommendedRows; }

    unsigned cols(unsigned recommendedCols) const
    { return recommendedCols; }

private:
    T_numtype value_;
};

BZ_NAMESPACE_END

#include <blitz/matref.h>
#include <blitz/matbops.h>
#include <blitz/matuops.h>

#endif // BZ_MATEXPR_H
