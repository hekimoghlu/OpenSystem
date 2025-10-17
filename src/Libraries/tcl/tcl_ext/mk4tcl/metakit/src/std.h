/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 2, 2025.
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

// std.h --
// $Id: std.h 1230 2007-03-09 15:58:53Z jcw $
// This is part of Metakit, see http://www.equi4.com/metakit.html

/** @file
 * Configuration header for STL-based builds
 */

#define q4_STD 1

#include "mk4str.h"

#include <vector>

/////////////////////////////////////////////////////////////////////////////

template <class T> class c4_ArrayT {
#if defined(_MSC_VER) || defined(__BORLANDC__)
    d4_std::vector<T, d4_std::allocator<T> > _vector;
#else 
    d4_std::vector<T, d4_std::alloc> _vector;
#endif 

  public:
    c4_ArrayT(){}
    ~c4_ArrayT(){}

    int GetSize()const {
        return _vector.size();
    }
    void SetSize(int nNewSize, int =  - 1) {
        _vector.resize(nNewSize);
    }

    T GetAt(int nIndex)const {
        return _vector[nIndex];
    }
    T &ElementAt(int nIndex) {
        return _vector[nIndex];
    }

    void SetAt(int nIndex, const T &newElement) {
        _vector[nIndex] = newElement;
    }

    int Add(const T &newElement) {
        int n = _vector.size();
        _vector.push_back(newElement);
        return n;
    }

    void InsertAt(int nIndex, const T &newElement, int nCount = 1) {
        _vector.insert(&_vector[nIndex], nCount, newElement);
    }

    void RemoveAt(int nIndex, int nCount = 1) {
        _vector.erase(&_vector[nIndex], &_vector[nIndex + nCount]);
    }
};

typedef c4_ArrayT < t4_i32 > c4_DWordArray;
typedef c4_ArrayT < void * > c4_PtrArray;
typedef c4_ArrayT < c4_String > c4_StringArray;

/////////////////////////////////////////////////////////////////////////////
