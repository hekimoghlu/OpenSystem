/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 8, 2025.
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

// univ.h --
// $Id: univ.h 1230 2007-03-09 15:58:53Z jcw $
// This is part of Metakit, the homepage is http://www.equi4.com/metakit.html

/** @file
 * Definition of the container classes
 */

#define q4_UNIV 1

#include "mk4str.h"

/////////////////////////////////////////////////////////////////////////////

class c4_BaseArray {
  public:
    c4_BaseArray();
    ~c4_BaseArray();

    int GetLength()const;
    void SetLength(int nNewSize);

    const void *GetData(int nIndex)const;
    void *GetData(int nIndex);

    void Grow(int nIndex);

    void InsertAt(int nIndex, int nCount);
    void RemoveAt(int nIndex, int nCount);

  private:
    char *_data;
    int _size;
    //  char _buffer[4];
};

class c4_PtrArray {
  public:
    c4_PtrArray();
    ~c4_PtrArray();

    int GetSize()const;
    void SetSize(int nNewSize, int nGrowBy =  - 1);

    void *GetAt(int nIndex)const;
    void SetAt(int nIndex, const void *newElement);
    void * &ElementAt(int nIndex);

    int Add(void *newElement);

    void InsertAt(int nIndex, void *newElement, int nCount = 1);
    void RemoveAt(int nIndex, int nCount = 1);

  private:
    static int Off(int n_);

    c4_BaseArray _vector;
};

class c4_DWordArray {
  public:
    c4_DWordArray();
    ~c4_DWordArray();

    int GetSize()const;
    void SetSize(int nNewSize, int nGrowBy =  - 1);

    t4_i32 GetAt(int nIndex)const;
    void SetAt(int nIndex, t4_i32 newElement);
    t4_i32 &ElementAt(int nIndex);

    int Add(t4_i32 newElement);

    void InsertAt(int nIndex, t4_i32 newElement, int nCount = 1);
    void RemoveAt(int nIndex, int nCount = 1);

  private:
    static int Off(int n_);

    c4_BaseArray _vector;
};

class c4_StringArray {
  public:
    c4_StringArray();
    ~c4_StringArray();

    int GetSize()const;
    void SetSize(int nNewSize, int nGrowBy =  - 1);

    const char *GetAt(int nIndex)const;
    void SetAt(int nIndex, const char *newElement);
    //  c4_String& ElementAt(int nIndex);

    int Add(const char *newElement);

    void InsertAt(int nIndex, const char *newElement, int nCount = 1);
    void RemoveAt(int nIndex, int nCount = 1);

  private:
    c4_PtrArray _ptrs;
};

/////////////////////////////////////////////////////////////////////////////

#if q4_INLINE
#include "univ.inl"
#endif 

/////////////////////////////////////////////////////////////////////////////
