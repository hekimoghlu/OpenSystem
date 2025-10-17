/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 11, 2022.
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
#ifndef BZ_ARRAYCYCLE_CC
#define BZ_ARRAYCYCLE_CC

#ifndef BZ_ARRAY_H
 #error <blitz/array/cycle.cc> must be included via <blitz/array.h>
#endif

BZ_NAMESPACE(blitz)

template<typename T_numtype, int N_rank>
void cycleArrays(Array<T_numtype, N_rank>& a, Array<T_numtype, N_rank>& b)
{
    Array<T_numtype, N_rank> tmp(a);
    a.reference(b);
    b.reference(tmp);
}

template<typename T_numtype, int N_rank>
void cycleArrays(Array<T_numtype, N_rank>& a, Array<T_numtype, N_rank>& b,
    Array<T_numtype, N_rank>& c)
{
    Array<T_numtype, N_rank> tmp(a);
    a.reference(b);
    b.reference(c);
    c.reference(tmp);
}

template<typename T_numtype, int N_rank>
void cycleArrays(Array<T_numtype, N_rank>& a, Array<T_numtype, N_rank>& b,
    Array<T_numtype, N_rank>& c, Array<T_numtype, N_rank>& d)
{
    Array<T_numtype, N_rank> tmp(a);
    a.reference(b);
    b.reference(c);
    c.reference(d);
    d.reference(tmp);
}

template<typename T_numtype, int N_rank>
void cycleArrays(Array<T_numtype, N_rank>& a, Array<T_numtype, N_rank>& b,
    Array<T_numtype, N_rank>& c, Array<T_numtype, N_rank>& d,
    Array<T_numtype, N_rank>& e)
{
    Array<T_numtype, N_rank> tmp(a);
    a.reference(b);
    b.reference(c);
    c.reference(d);
    d.reference(e);
    e.reference(tmp);
}

BZ_NAMESPACE_END

#endif // BZ_ARRAYCYCLE_CC
