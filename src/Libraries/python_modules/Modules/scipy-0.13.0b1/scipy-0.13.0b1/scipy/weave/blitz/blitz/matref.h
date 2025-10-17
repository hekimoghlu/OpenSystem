/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 22, 2025.
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
#ifndef BZ_MATREF_H
#define BZ_MATREF_H

#ifndef BZ_MATEXPR_H
 #error <blitz/matref.h> must be included via <blitz/matexpr.h>
#endif // BZ_MATEXPR_H

BZ_NAMESPACE(blitz)

template<typename P_numtype, typename P_structure>
class _bz_MatrixRef {

public:
    typedef P_numtype T_numtype;

    _bz_MatrixRef(const Matrix<P_numtype, P_structure>& m)
        : matrix_(&m)
    { }

    T_numtype operator()(unsigned i, unsigned j) const
    { return (*matrix_)(i,j); }

    unsigned rows(unsigned) const
    { return matrix_->rows(); }

    unsigned cols(unsigned) const
    { return matrix_->cols(); }

private:
    _bz_MatrixRef() { } 

    const Matrix<P_numtype, P_structure>* matrix_;
};

BZ_NAMESPACE_END

#endif // BZ_MATREF_H
