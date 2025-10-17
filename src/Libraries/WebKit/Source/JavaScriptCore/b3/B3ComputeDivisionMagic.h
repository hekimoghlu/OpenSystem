/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 14, 2023.
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
#pragma once

#if ENABLE(B3_JIT)

namespace JSC { namespace B3 {

template<typename T>
struct DivisionMagic {
    T magicMultiplier;
    unsigned shift;
};

// This contains code taken from LLVM's APInt::magic(). It's modestly adapted to our style, but
// not completely, to make it easier to apply their changes in the future.
template<typename T>
DivisionMagic<T> computeDivisionMagic(T divisor)
{
    typedef typename std::make_unsigned<T>::type UnsignedT;
    UnsignedT d = divisor;
    unsigned p;
    UnsignedT ad, anc, delta, q1, r1, q2, r2, t;
    UnsignedT signedMin = static_cast<UnsignedT>(std::numeric_limits<T>::min());
    DivisionMagic<T> mag;
    unsigned bitWidth = sizeof(divisor) * 8;

    // This code doesn't like to think of signedness as a type. Instead it likes to think that
    // operations have signedness. This is how we generally do it in B3 as well. For this reason,
    // we cast all the operated values once to unsigned. And later, we convert it to signed.
    // Only `divisor` have signedness here.

    ad = divisor < 0 ? -divisor : divisor; // -(signed min value) < signed max value. So there is no loss.
    t = signedMin + (d >> (bitWidth - 1));
    anc = t - 1 - (t % ad);   // absolute value of nc
    p = bitWidth - 1;    // initialize p
    q1 = signedMin / anc;   // initialize q1 = 2p/abs(nc)
    r1 = signedMin - q1*anc;    // initialize r1 = rem(2p,abs(nc))
    q2 = signedMin / ad;    // initialize q2 = 2p/abs(d)
    r2 = signedMin - q2*ad;     // initialize r2 = rem(2p,abs(d))
    do {
        p = p + 1;
        q1 = q1 << 1;          // update q1 = 2p/abs(nc)
        r1 = r1 << 1;          // update r1 = rem(2p/abs(nc))
        if (r1 >= anc) {  // must be unsigned comparison
            q1 = q1 + 1;
            r1 = r1 - anc;
        }
        q2 = q2 << 1;          // update q2 = 2p/abs(d)
        r2 = r2 << 1;          // update r2 = rem(2p/abs(d))
        if (r2 >= ad) {   // must be unsigned comparison
            q2 = q2 + 1;
            r2 = r2 - ad;
        }
        delta = ad - r2;
    } while (q1 < delta || (q1 == delta && r1 == 0));

    mag.magicMultiplier = q2 + 1;
    if (divisor < 0)
        mag.magicMultiplier = -mag.magicMultiplier;   // resulting magic number
    mag.shift = p - bitWidth;          // resulting shift

    return mag;
}

} } // namespace JSC::B3

#endif // ENABLE(B3_JIT)
