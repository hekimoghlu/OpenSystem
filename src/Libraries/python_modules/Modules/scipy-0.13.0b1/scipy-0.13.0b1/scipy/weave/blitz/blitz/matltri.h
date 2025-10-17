/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 18, 2023.
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
#ifndef BZ_MATLTRI_H
#define BZ_MATLTRI_H

#ifndef BZ_MSTRUCT_H
 #error <blitz/matltri.h> must be included via <blitz/mstruct.h>
#endif

BZ_NAMESPACE(blitz)

// Lower triangular, row major ordering
// [ 0 . . . ]
// [ 1 2 . . ]
// [ 3 4 5 . ]
// [ 6 7 8 9 ]

class LowerTriangularIterator {
public:
    LowerTriangularIterator(unsigned rows, unsigned cols)
    {
        BZPRECONDITION(rows == cols);
        size_ = rows;
        good_ = true;
        offset_ = 0;
        i_ = 0;
        j_ = 0;
    }
   
    operator bool() const { return good_; }

    void operator++()
    {
        BZPRECONDITION(good_);
        ++offset_;
        ++j_;
        if (j_ > i_)
        {
            j_ = 0;
            ++i_;
            if (i_ == size_)
                good_ = false;
        }
    }

    unsigned row() const
    { return i_; }

    unsigned col() const
    { return j_; }

    unsigned offset() const
    { return offset_; }

protected:
    unsigned size_;
    bool good_;
    unsigned offset_;
    unsigned i_, j_;
};

class LowerTriangular : public MatrixStructure {

public:
    typedef LowerTriangularIterator T_iterator;

    LowerTriangular()
        : size_(0)
    { }

    LowerTriangular(unsigned size)
        : size_(size)
    { }

    LowerTriangular(unsigned rows, unsigned cols)
        : size_(rows)
    {
        BZPRECONDITION(rows == cols);
    }

    unsigned columns() const
    { return size_; }

    unsigned coordToOffset(unsigned i, unsigned j) const
    {
        BZPRECONDITION(inRange(i,j));
        BZPRECONDITION(i >= j);
        return i*(i+1)/2 + j;
    }

    unsigned firstInRow(unsigned i) const
    { return 0; }

    template<typename T_numtype>
    T_numtype get(const T_numtype * restrict data,
        unsigned i, unsigned j) const
    {
        BZPRECONDITION(inRange(i,j));
        if (i >= j)
            return data[coordToOffset(i,j)];
        else
            return ZeroElement<T_numtype>::zero();
    }

    template<typename T_numtype>
    T_numtype& get(T_numtype * restrict data, unsigned i, unsigned j)
    {
        BZPRECONDITION(inRange(i,j));
        if (i >= j)
            return data[coordToOffset(i,j)];
        else
            return ZeroElement<T_numtype>::zero();
    }

    unsigned lastInRow(unsigned i) const
    { return i; }

    unsigned firstInCol(unsigned j) const
    { return j; }

    unsigned lastInCol(unsigned j) const
    { return size_ - 1; }

    bool inRange(unsigned i, unsigned j) const
    {
        return (i < size_) && (j < size_);
    }

    unsigned numElements() const
    { return size_ * (size_ + 1) / 2; }

    unsigned rows() const
    { return size_; }

    void resize(unsigned size)
    {
        size_ = size;
    }

    void resize(unsigned rows, unsigned cols)
    {
        BZPRECONDITION(rows == cols);
        size_  = rows;
    }

private:
    unsigned size_;
};

BZ_NAMESPACE_END

#endif // BZ_MATLTRI_H

