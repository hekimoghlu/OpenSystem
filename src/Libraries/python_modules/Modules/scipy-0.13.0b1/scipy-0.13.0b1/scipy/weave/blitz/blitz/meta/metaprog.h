/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 31, 2022.
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
 * blitz/meta/metaprog.h   Useful metaprogram declarations
 *
 * $Id: metaprog.h 1413 2005-11-01 22:04:15Z cookedm $
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
 ***************************************************************************/

#ifndef BZ_META_METAPROG_H
#define BZ_META_METAPROG_H

BZ_NAMESPACE(blitz)

// Null Operand

class _bz_meta_nullOperand {
public:
    _bz_meta_nullOperand() { }
};

template<typename T> inline T operator+(const T& a, _bz_meta_nullOperand)
{ return a; }
template<typename T> inline T operator*(const T& a, _bz_meta_nullOperand)
{ return a; }

// MetaMax

template<int N1, int N2>
class _bz_meta_max {
public:
    static const int max = (N1 > N2) ? N1 : N2;
};

// MetaMin

template<int N1, int N2>
class _bz_meta_min {
public:
    static const int min = (N1 < N2) ? N1 : N2;
};

BZ_NAMESPACE_END 

#endif // BZ_META_METAPROG_H
