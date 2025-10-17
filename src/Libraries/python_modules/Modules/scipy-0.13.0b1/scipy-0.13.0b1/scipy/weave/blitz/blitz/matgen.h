/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 9, 2024.
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
#ifndef BZ_MATGEN_H
#define BZ_MATGEN_H

#ifndef BZ_MSTRUCT_H
 #error <blitz/matgen.h> must be included via <blitz/mstruct.h>
#endif // BZ_MSTRUCT_H

BZ_NAMESPACE(blitz)

class GeneralMatrix : public AsymmetricMatrix {

public:
    GeneralMatrix()
    { }

    GeneralMatrix(unsigned rows, unsigned cols)
        : AsymmetricMatrix(rows, cols)
    {
    }

    unsigned firstInRow(unsigned i) const
    { return 0; }

    unsigned lastInRow(unsigned i) const
    { return cols_ - 1; }

    unsigned firstInCol(unsigned j) const
    { return 0; }

    unsigned lastInCol(unsigned j) const
    { return rows_ - 1; }

    unsigned numElements() const
    { return rows_ * cols_; }
};

class GeneralIterator {
public:
    GeneralIterator(unsigned rows, unsigned cols)
    {
        rows_ = rows;
        cols_ = cols;
        i_ = 0;
        j_ = 0;
        offset_ = 0;
        good_ = true;
    }

    unsigned offset() const { return offset_; }
    operator bool()   const { return good_; }
    unsigned row()    const { return i_; }
    unsigned col()    const { return j_; }
 
protected:
    unsigned rows_, cols_;
    unsigned offset_;
    unsigned i_, j_;
    bool     good_;
};

class RowMajorIterator : public GeneralIterator {
public:
    RowMajorIterator(unsigned rows, unsigned cols)
        : GeneralIterator(rows, cols)
    { }

    void operator++()
    {
        ++offset_;
        ++j_;
        if (j_ == cols_)
        {
            j_ = 0;
            ++i_;
            if (i_ == rows_)
                good_ = false;
        }
    }
};

class RowMajor : public GeneralMatrix {

public:
    typedef RowMajorIterator T_iterator;

    RowMajor()
    { }

    RowMajor(unsigned rows, unsigned cols)
        : GeneralMatrix(rows, cols)
    { }

    unsigned coordToOffset(unsigned i, unsigned j) const
    {
        return i*cols_+j;
    }

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
};

class ColumnMajorIterator : public GeneralIterator {
public:
    ColumnMajorIterator(unsigned rows, unsigned cols)
        : GeneralIterator(rows, cols)
    {
    }

    void operator++()
    {
        ++offset_;
        ++i_;
        if (i_ == rows_)
        {
            i_ = 0;
            ++j_;
            if (j_ == cols_)
                good_ = false;
        }
    }
};

class ColumnMajor : public GeneralMatrix {

public:
    ColumnMajor()
    { }

    ColumnMajor(unsigned rows, unsigned cols)
        : GeneralMatrix(rows, cols)
    { }

    unsigned coordToOffset(unsigned i, unsigned j) const
    {
        return j*rows_ + i;
    }

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
};

BZ_NAMESPACE_END

#endif // BZ_MATGEN_H

