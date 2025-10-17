/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 30, 2022.
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
#ifndef BZ_MATTOEP_H
#define BZ_MATTOEP_H

#ifndef BZ_MSTRUCT_H
 #error <blitz/mattoep.h> must be included via <blitz/mstruct.h>
#endif

BZ_NAMESPACE(blitz)

// Toeplitz matrix
// [ 0 1 2 3 ]
// [ 1 2 3 4 ]
// [ 2 3 4 5 ]
// [ 3 4 5 6 ]

class ToeplitzIterator {
public:
    ToeplitzIterator(unsigned rows, unsigned cols)
    {
        rows_ = rows;
        cols_ = cols;
        i_ = 0;
        j_ = 0;
        good_ = true;
        offset_ = 0;
    }

    operator bool() const { return good_; }

    void operator++()
    {
        ++offset_;
        if (i_ < rows_ - 1)
            ++i_;
        else if (j_ < cols_ - 1)
            ++j_;
        else
            good_ = false;
    }

    unsigned row() const
    { return i_; }

    unsigned col() const
    { return j_; }

    unsigned offset() const
    { return offset_; }

protected:
    unsigned offset_;
    unsigned i_, j_;
    unsigned rows_, cols_;
    bool     good_;
};

class Toeplitz : public GeneralMatrix {

public:
    typedef ToeplitzIterator T_iterator;

    Toeplitz()
        : rows_(0), cols_(0)
    { }

    Toeplitz(unsigned rows, unsigned cols)
        : rows_(rows), cols_(cols)
    { }

    unsigned columns() const
    { return cols_; }

    unsigned coordToOffset(unsigned i, unsigned j) const
    {
        BZPRECONDITION(inRange(i,j));
        return i + j;
    }

    unsigned firstInRow(unsigned i) const
    { return 0; }

    template<typename T_numtype>
    T_numtype get(const T_numtype * restrict data,
        unsigned i, unsigned j) const
    {
        BZPRECONDITION(inRange(i,j));
        return data[coordToOffset(i,j)];
    }

    template<typename T_numtype>
    T_numtype& get(T_numtype * restrict data, unsigned i, unsigned j)
    {
        BZPRECONDITION(inRange(i,j));
        return data[coordToOffset(i,j)];
    }

    unsigned lastInRow(const unsigned)  const { return cols_ - 1; }
    unsigned firstInCol(const unsigned) const { return 0; }
    unsigned lastInCol(const unsigned)  const { return rows_ - 1; }

    bool inRange(const unsigned i,const unsigned j) const { return (i<rows_) && (j<cols_); }

    unsigned numElements() const { return rows_ + cols_ - 1; }

    unsigned rows() const { return rows_; }

    void resize(const unsigned rows,const unsigned cols) {
        rows_ = rows;
        cols_ = cols;
    }

private:
    unsigned rows_, cols_;
};

BZ_NAMESPACE_END

#endif // BZ_MATSYMM_H

