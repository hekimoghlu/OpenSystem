/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 4, 2023.
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
#ifndef BZ_RAND_UNIFORM_H
#define BZ_RAND_UNIFORM_H

#ifndef BZ_RANDOM_H
 #include <blitz/random.h>
#endif

BZ_NAMESPACE(blitz)

class Uniform {

public:
    typedef double T_numtype;

    Uniform(double low = 0.0, double high = 1.0, double = 0.0)
        : low_(low), length_(high-low)
    { 
        BZPRECONDITION(sizeof(int) >= 4);   // Need 32 bit integers!

        seed[0] = 24;       // All seeds in the range [0,4095]
        seed[1] = 711;
        seed[2] = 3;
        seed[3] = 3721;     // The last seed must be odd
    }

    void randomize() 
    { 
        BZ_NOT_IMPLEMENTED();            // NEEDS_WORK

        BZPOSTCONDITION(seed[3] % 2 == 1);
    }
  
    // I'm trying to avoid having a compiled 
    // portion of the library, so this is inline until I
    // figure out a better way to do this or I change my mind.
    // -- TV
    // NEEDS_WORK
    double random()
    { 
        BZPRECONDITION(seed[3] % 2 == 1);

        int it0, it1, it2, it3;
        it3 = seed[3] * 2549;
        it2 = it3 / 4096;
        it3 -= it2 << 12;
        it2 += seed[2] * 2549 + seed[3] * 2508;
        it1 = it2 / 4096;
        it2 -= it1 << 12;
        it1 += seed[1] * 2549 + seed[2] * 2508 + seed[3] * 322;
        it0 = it1 / 4096;
        it1 -= it0 << 12;
        it0 += seed[0] * 2549 + seed[1] * 2508 + seed[2] * 322 + seed[3] * 494;
        it0 %= 4096;
        seed[0] = it0;
        seed[1] = it1;
        seed[2] = it2;
        seed[3] = it3;
      
        const double z = 1 / 4096.;
        return low_ + length_ * (it0 + (it1 + (it2 + it3 * z) * z) * z) * z;
    } 

    operator double() 
    { return random(); }

private:
    double low_, length_;

    int seed[4];
};

BZ_NAMESPACE_END

#endif // BZ_RAND_UNIFORM_H

