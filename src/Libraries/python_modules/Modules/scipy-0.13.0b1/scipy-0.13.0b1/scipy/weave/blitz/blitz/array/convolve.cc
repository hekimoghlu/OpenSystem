/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 15, 2024.
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
#ifndef BZ_ARRAY_CONVOLVE_CC
#define BZ_ARRAY_CONVOLVE_CC

BZ_NAMESPACE(blitz)

template<typename T>
Array<T,1> convolve(const Array<T,1>& B, const Array<T,1>& C)
{
    int Bl = B.lbound(0), Bh = B.ubound(0);
    int Cl = C.lbound(0), Ch = C.ubound(0);

    int lbound = Bl + Cl;
    int ubound = Bh + Ch;
    
    Array<T,1> A(Range(lbound,ubound));

    for (int i=lbound; i <= ubound; ++i)
    {
        int jl = i - Ch;
        if (jl < Bl)
            jl = Bl;

        int jh = i - Cl;
        if (jh > Bh)
            jh = Bh;

        T result = 0;
        for (int j=jl; j <= jh; ++j)
            result += B(j) * C(i-j);

        A(i) = result;
    }

    return A;
}

BZ_NAMESPACE_END

#endif // BZ_ARRAY_CONVOLVE_CC

