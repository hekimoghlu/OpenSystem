/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 29, 2025.
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
#ifndef BZ_ARRAYCOMPLEX_CC
#define BZ_ARRAYCOMPLEX_CC

// Special functions for complex arrays

#ifndef BZ_ARRAY_H
 #error <blitz/array/complex.cc> must be included via <blitz/array/array.h>
#endif

BZ_NAMESPACE(blitz)

#ifdef BZ_HAVE_COMPLEX

template<typename T_numtype, int N_rank>
inline Array<T_numtype, N_rank> real(const Array<complex<T_numtype>,N_rank>& A)
{
    return A.extractComponent(T_numtype(), 0, 2);
}

template<typename T_numtype, int N_rank>
inline Array<T_numtype, N_rank> imag(const Array<complex<T_numtype>,N_rank>& A)
{
    return A.extractComponent(T_numtype(), 1, 2);
}


#endif

BZ_NAMESPACE_END

#endif // BZ_ARRAYCOMPLEX_CC

