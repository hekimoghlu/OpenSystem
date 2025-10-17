/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 1, 2022.
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
 * blitz/array/slice.h    Helper classes for slicing arrays
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
 ****************************************************************************/
#ifndef BZ_ARRAYSLICE_H
#define BZ_ARRAYSLICE_H

#ifndef BZ_ARRAY_H
 #error <blitz/array/slice.h> must be included via <blitz/array.h>
#endif

BZ_NAMESPACE(blitz)

// Forward declaration
template<typename T, int N>
class Array;



class nilArraySection { };

template<typename T>
class ArraySectionInfo {
public:
    static const int isValidType = 0, rank = 0, isPick = 0;
};

template<>
class ArraySectionInfo<Range> {
public:
    static const int isValidType = 1, rank = 1, isPick = 0;
};

template<>
class ArraySectionInfo<int> {
public:
    static const int isValidType = 1, rank = 0, isPick = 0;
};

template<>
class ArraySectionInfo<nilArraySection> {
public:
    static const int isValidType = 1, rank = 0, isPick = 0;
};

template<typename T_numtype, typename T1, typename T2 = nilArraySection, 
    class T3 = nilArraySection, typename T4 = nilArraySection, 
    class T5 = nilArraySection, typename T6 = nilArraySection, 
    class T7 = nilArraySection, typename T8 = nilArraySection, 
    class T9 = nilArraySection, typename T10 = nilArraySection, 
    class T11 = nilArraySection>
class SliceInfo {
public:
    static const int 
        numValidTypes = ArraySectionInfo<T1>::isValidType
                      + ArraySectionInfo<T2>::isValidType
                      + ArraySectionInfo<T3>::isValidType
                      + ArraySectionInfo<T4>::isValidType
                      + ArraySectionInfo<T5>::isValidType
                      + ArraySectionInfo<T6>::isValidType
                      + ArraySectionInfo<T7>::isValidType
                      + ArraySectionInfo<T8>::isValidType
                      + ArraySectionInfo<T9>::isValidType
                      + ArraySectionInfo<T10>::isValidType
                      + ArraySectionInfo<T11>::isValidType,

        rank          = ArraySectionInfo<T1>::rank
                      + ArraySectionInfo<T2>::rank
                      + ArraySectionInfo<T3>::rank
                      + ArraySectionInfo<T4>::rank
                      + ArraySectionInfo<T5>::rank
                      + ArraySectionInfo<T6>::rank
                      + ArraySectionInfo<T7>::rank
                      + ArraySectionInfo<T8>::rank
                      + ArraySectionInfo<T9>::rank
                      + ArraySectionInfo<T10>::rank
                      + ArraySectionInfo<T11>::rank,

        isPick        = ArraySectionInfo<T1>::isPick
                      + ArraySectionInfo<T2>::isPick
                      + ArraySectionInfo<T3>::isPick
                      + ArraySectionInfo<T4>::isPick
                      + ArraySectionInfo<T5>::isPick
                      + ArraySectionInfo<T6>::isPick
                      + ArraySectionInfo<T7>::isPick
                      + ArraySectionInfo<T8>::isPick
                      + ArraySectionInfo<T9>::isPick
                      + ArraySectionInfo<T10>::isPick
                      + ArraySectionInfo<T11>::isPick;

    typedef Array<T_numtype,numValidTypes> T_array;
    typedef Array<T_numtype,rank> T_slice;
};

BZ_NAMESPACE_END

#endif // BZ_ARRAYSLICE_H
