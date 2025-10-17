/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 1, 2022.
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

// -*- C++ -*-
/***************************************************************************
 * blitz/array/asexpr.h  Declaration of the asExpr helper functions
 *
 * Copyright (C) 1997-2001 Todd Veldhuizen <tveldhui@oonumerics.org>
 *
 * This code was relicensed under the modified BSD license for use in SciPy
 * by Todd Veldhuizen (see LICENSE.txt in the weave directory).
 *
 *
 * Suggestions:          blitz-dev@oonumerics.org
 * Bugs:                 blitz-bugs@oonumerics.org
 *
 * For more information, please see the Blitz++ Home Page:
 *    http://oonumerics.org/blitz/
 *
 ***************************************************************************/
#ifndef BZ_ARRAYASEXPR_H
#define BZ_ARRAYASEXPR_H

#ifndef BZ_ARRAY_H
 #error <blitz/array/asexpr.h> must be included via <blitz/array.h>
#endif

BZ_NAMESPACE(blitz)

// The traits class asExpr converts arbitrary things to
// expression templatable operands.

//  Default to scalar.

template <typename T>
struct asExpr {
    typedef _bz_ArrayExprConstant<T> T_expr;
    static T_expr getExpr(const T& x) { return T_expr(x); }
};

//  Already an expression template term

template <typename T>
struct asExpr<_bz_ArrayExpr<T> > {
    typedef _bz_ArrayExpr<T> T_expr;
    static const T_expr& getExpr(const T_expr& x) { return x; }
};

//  An array operand

template <typename T,int N>
struct asExpr<Array<T,N> > {
    typedef FastArrayIterator<T,N> T_expr;
    static T_expr getExpr(const Array<T,N>& x) { return x.beginFast(); }
};

//  Index placeholder

template <int N>
struct asExpr<IndexPlaceholder<N> > {
    typedef IndexPlaceholder<N> T_expr;
    static T_expr getExpr(T_expr x) { return x; }
};

#ifdef BZ_HAVE_TEMPLATES_AS_TEMPLATE_ARGUMENTS

//  A traits class that provides the return type of a binary operation.

template <template <typename T1> class OP, typename O1>
struct BzUnaryExprResult {
    typedef _bz_ArrayExpr<_bz_ArrayExprUnaryOp<
        typename asExpr<O1>::T_expr,
        OP<typename asExpr<O1>::T_expr::T_numtype> > > T_result;
};

template <template <typename T1, typename T2> class OP,
          typename O1, typename O2>
struct BzBinaryExprResult {
    typedef _bz_ArrayExpr<_bz_ArrayExprBinaryOp<
        typename asExpr<O1>::T_expr,
        typename asExpr<O2>::T_expr,
        OP<typename asExpr<O1>::T_expr::T_numtype,
           typename asExpr<O2>::T_expr::T_numtype> > > T_result;
};

template <template <typename T1, typename T2, typename T3> class OP,
          typename O1, typename O2, typename O3>
struct BzTernaryExprResult {
    typedef _bz_ArrayExpr<_bz_ArrayExprTernaryOp<
        typename asExpr<O1>::T_expr,
        typename asExpr<O2>::T_expr,
        typename asExpr<O3>::T_expr,
        OP<typename asExpr<O1>::T_expr::T_numtype,
           typename asExpr<O2>::T_expr::T_numtype,
           typename asExpr<O3>::T_expr::T_numtype> > > T_result;
};

#endif /* BZ_HAVE_TEMPLATES_AS_TEMPLATE_ARGUMENTS */

BZ_NAMESPACE_END

#endif
