/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 29, 2023.
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
#ifndef BZ_TINYVECIO_CC
#define BZ_TINYVECIO_CC

#ifndef BZ_TINYVEC_H
 #include <blitz/tinyvec.h>
#endif

BZ_NAMESPACE(blitz)

// NEEDS_WORK

template<typename P_numtype, int N_length>
ostream& operator<<(ostream& os, const TinyVector<P_numtype, N_length>& x)
{
    os << N_length << " [ ";
    for (int i=0; i < N_length; ++i)
    {
        os << setw(10) << x[i];
        if (!((i+1)%7))
            os << endl << "  ";
    }
    os << " ]";
    return os;
}

// Input of tinyvec contribute by Adam Levar <adaml@mcneilhouse.com>
template <typename T_numtype, int N_length>
istream& operator>>(istream& is, TinyVector<T_numtype, N_length>& x)
{
    int length;
    char sep;
             
    is >> length;
    is >> sep;
    BZPRECHECK(sep == '[', "Format error while scanning input array"
        << endl << " (expected '[' before beginning of array data)");

    BZPRECHECK(length == N_length, "Size mismatch");                    
    for (int i = 0; i < N_length; ++i)
    {
        BZPRECHECK(!is.bad(), "Premature end of input while scanning array");
        is >> x(i);
    }
    is >> sep;
    BZPRECHECK(sep == ']', "Format error while scanning input array"
       << endl << " (expected ']' after end of array data)");
    
    return is;
}

BZ_NAMESPACE_END

#endif // BZ_TINYVECIO_CC
