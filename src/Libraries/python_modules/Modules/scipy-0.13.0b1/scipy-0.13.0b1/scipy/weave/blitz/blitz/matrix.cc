/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 24, 2022.
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
#ifndef BZ_MATRIX_CC
#define BZ_MATRIX_CC

#ifndef BZ_MATRIX_H
 #include <blitz/matrix.h>
#endif

BZ_NAMESPACE(blitz)

// Matrix expression operand
template<typename P_numtype, typename P_structure> template<typename P_expr>
Matrix<P_numtype, P_structure>& 
Matrix<P_numtype, P_structure>::operator=(_bz_MatExpr<P_expr> expr)
{
    // Check for compatible structures.

    // Fast evaluation (compatible structures)
    // (not implemented)

    // Slow evaluation
    _bz_typename P_structure::T_iterator iter(rows(), cols());
    while (iter)
    {
        data_[iter.offset()] = expr(iter.row(), iter.col());
        ++iter;
    }

    return *this;
}

template<typename P_numtype, typename P_structure>
ostream& operator<<(ostream& os, const Matrix<P_numtype, P_structure>& matrix)
{
    os << "[ ";
    for (int i=0; i < matrix.rows(); ++i)
    {
        for (int j=0; j < matrix.columns(); ++j)
        {
            os << setw(10) << matrix(i,j);
            if ((!((j+1)%7)) && (j < matrix.cols()-1))
                os << endl << "         ...";
        }
        if (i != matrix.rows() - 1)
            os << endl  << "  ";
    }
    os << " ]";
    return os;
}

BZ_NAMESPACE_END

#endif // BZ_MATRIX_CC
