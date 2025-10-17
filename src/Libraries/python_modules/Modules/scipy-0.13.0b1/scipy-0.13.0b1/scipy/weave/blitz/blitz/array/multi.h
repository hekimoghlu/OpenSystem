/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 12, 2022.
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
 * blitz/array/multi.h  Support for multicomponent arrays
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
#ifndef BZ_ARRAYMULTI_H
#define BZ_ARRAYMULTI_H

#ifndef BZ_ARRAY_H
 #error <blitz/array/multi.h> must be included via <blitz/array.h>
#endif

BZ_NAMESPACE(blitz)

/*
 * The multicomponent_traits class provides a mapping from multicomponent
 * tuples to the element type they contain.  For example:
 *
 * multicomponent_traits<complex<float> >::T_numtype is float,
 * multicomponent_traits<TinyVector<int,3> >::T_numtype is int.
 *
 * This is used to support Array<T,N>::operator[], which extracts components
 * from a multicomponent array.
 */

// By default, produce a harmless component type, and zero components.
template<typename T_component>
struct multicomponent_traits {
    typedef T_component T_element;
    static const int numComponents = 0;
};

// TinyVector
template<typename T_numtype, int N_rank>
struct multicomponent_traits<TinyVector<T_numtype,N_rank> > {
    typedef T_numtype T_element;
    static const int numComponents = N_rank;
};

#ifdef BZ_HAVE_COMPLEX
// complex<T>
template<typename T>
struct multicomponent_traits<complex<T> > {
    typedef T T_element;
    static const int numComponents = 2;
};
#endif

// This macro is provided so that users can register their own
// multicomponent types.

#define BZ_DECLARE_MULTICOMPONENT_TYPE(T_tuple,T,N)          \
  BZ_NAMESPACE(blitz)                                        \
  template<>                                                 \
  struct multicomponent_traits<T_tuple > {                   \
    typedef T T_element;                                     \
    static const int numComponents = N;                      \
  };                                                         \
  BZ_NAMESPACE_END

BZ_NAMESPACE_END

#endif // BZ_ARRAYMULTI_H
