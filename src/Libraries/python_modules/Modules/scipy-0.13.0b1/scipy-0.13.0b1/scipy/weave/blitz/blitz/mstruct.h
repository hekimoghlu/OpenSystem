/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 16, 2025.
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
#ifndef BZ_MSTRUCT_H
#define BZ_MSTRUCT_H

#ifndef BZ_BLITZ_H
 #include <blitz/blitz.h>
#endif

#ifndef BZ_ZERO_H
 #include <blitz/zero.h>
#endif

/*
 * Each matrix structure class encapsulates a storage format for matrix
 * data.  It is responsible for:
 * - Storing the size of the matrix
 * - Calculating how many unique elements the matrix will have
 * - Mapping indices (i,j) onto memory locations
 * - Performing any sign reversals or conjugations when matrix
 *   elements are retrieved (e.g. in a Hermitian or Skew symmetric
 *   matrix)
 *
 * Every matrix structure class must provide these methods:
 *
 * ctor()
 * ctor(unsigned rows, unsigned cols)
 * unsigned columns() const;
 * unsigned cols()    const;
 * unsigned firstInRow() const;
 * unsigned firstInCol() const;
 * template<typename T> T& get(T* data, unsigned i, unsigned j);
 * template<typename T> T  get(const T* data, unsigned i, unsigned j) const;
 * bool inRange(unsigned i, unsigned j) const
 * unsigned lastInRow() const;
 * unsigned lastInCol() const;
 * unsigned numElements() const;
 * void resize(unsigned rows, unsigned cols);
 * unsigned rows()    const;
 *
 * Each matrix structure class must declare a public type
 * T_iterator which is an iterator to scan through the unique
 * entries of the matrix.  The iterator class must provide
 * these methods:
 *
 * ctor(unsigned rows, unsigned cols)
 * unsigned offset() const
 * operator bool() const
 * unsigned row() const
 * unsigned col() const
 */

BZ_NAMESPACE(blitz)

class MatrixStructure { };

class AsymmetricMatrix : public MatrixStructure {
public:
    AsymmetricMatrix()
        : rows_(0), cols_(0)
    { }

    AsymmetricMatrix(unsigned rows, unsigned cols)
        : rows_(rows), cols_(cols)
    { }

    unsigned columns() const { return cols_; }

    unsigned cols() const { return cols_; }

    bool inRange(const unsigned i,const unsigned j) const {
        return (i<rows_) && (j<cols_);
    }

    void resize(unsigned rows, unsigned cols) {
        rows_ = rows;
        cols_ = cols;
    }

    unsigned rows() const { return rows_; }

protected:
    unsigned rows_, cols_;
};

// Still to be implemented:
// SkewSymmetric
// Hermitian
// Tridiagonal
// Banded<L,H>
// Upper bidiagonal
// Lower bidiagonal
// Upper Hessenberg
// Lower Hessenberg

BZ_NAMESPACE_END

#include <blitz/matgen.h>         // RowMajor and ColumnMajor general matrices
#include <blitz/matsymm.h>        // Symmetric
#include <blitz/matdiag.h>        // Diagonal
#include <blitz/mattoep.h>        // Toeplitz
#include <blitz/matltri.h>        // Lower triangular
#include <blitz/matutri.h>        // Upper triangular

#endif // BZ_MSTRUCT_H
