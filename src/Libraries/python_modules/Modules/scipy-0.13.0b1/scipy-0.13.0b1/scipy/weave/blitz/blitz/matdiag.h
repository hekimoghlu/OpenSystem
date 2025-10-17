/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 25, 2023.
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
#ifndef BZ_MATDIAG_H
#define BZ_MATDIAG_H

#ifndef BZ_MSTRUCT_H
 #error <blitz/matdiag.h> must be included via <blitz/mstruct.h>
#endif

BZ_NAMESPACE(blitz)

// Diagonal matrix
// [ 0 . . . ]
// [ . 1 . . ]
// [ . . 2 . ]
// [ . . . 3 ]

class DiagonalIterator {
public:
    DiagonalIterator(const unsigned rows,const unsigned cols) {
        BZPRECONDITION(rows==cols);
        size_ = rows;
        i_ = 0;
    }

    operator bool() const { return i_ < size_; }

    void operator++() { ++i_; }

    unsigned row()    const { return i_; }
    unsigned col()    const { return i_; }
    unsigned offset() const { return i_; }

protected:
    unsigned i_, size_;
};

class Diagonal : public MatrixStructure {
public:
    typedef DiagonalIterator T_iterator;

    Diagonal(): size_(0) { }

    Diagonal(const unsigned size): size_(size) { }

    Diagonal(const unsigned rows,const unsigned cols): size_(rows) {
        BZPRECONDITION(rows == cols);
    }

    unsigned columns() const { return size_; }

    unsigned coordToOffset(const unsigned i,const unsigned j) const
    {
        BZPRECONDITION(inRange(i,j));
        BZPRECONDITION(i == j);
        return i;
    }

    unsigned firstInRow(const unsigned i) const { return i; }

    template<typename T_numtype>
    T_numtype get(const T_numtype * restrict data,const unsigned i,const unsigned j) const
    {
        BZPRECONDITION(inRange(i,j));
        return (i==j) ? data[coordToOffset(i,j)] : ZeroElement<T_numtype>::zero();
    }

    template<typename T_numtype>
    T_numtype& get(T_numtype * restrict data,const unsigned i,const unsigned j) {
        BZPRECONDITION(inRange(i,j));
        return (i==j) ? data[coordToOffset(i,j)] : ZeroElement<T_numtype>::zero();
    }

    unsigned lastInRow(const unsigned i)  const { return i; }
    unsigned firstInCol(const unsigned j) const { return j; }
    unsigned lastInCol(const unsigned j)  const { return j; }

    bool inRange(const unsigned i,const unsigned j) const {
        return (i < size_) && (j < size_);
    }

    unsigned numElements() const { return size_; }
    unsigned rows()        const { return size_; }

    void resize(const unsigned size) { size_ = size; }

    void resize(const unsigned rows,const unsigned cols) {
        BZPRECONDITION(rows == cols);
        size_  = rows;
    }

private:
    unsigned size_;
};

BZ_NAMESPACE_END

#endif // BZ_MATSYMM_H
