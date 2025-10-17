/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 11, 2024.
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
#ifndef BZ_RAND_NORMAL_H
#define BZ_RAND_NORMAL_H

#ifndef BZ_RANDOM_H
 #include <blitz/random.h>
#endif

#ifndef BZ_RAND_UNIFORM_H
 #include <blitz/rand-uniform.h>
#endif

#include <math.h>

BZ_NAMESPACE(blitz)

template<typename P_uniform BZ_TEMPLATE_DEFAULT(Uniform)>
class Normal {

public:
    typedef double T_numtype;

    Normal(double mean = 0.0, double variance = 1.0, double = 0.0)
        : mean_(mean), sigma_(::sqrt(variance))
    { 
    }

    void randomize() 
    { 
        uniform_.randomize();
    }
  
    double random()
    { 
        double u, v;

        do {
            u = uniform_.random();
            v = uniform_.random();    
        } while (v == 0);

        return mean_ + sigma_ * ::sqrt(-2*::log(v)) * ::cos(M_PI * (2*u - 1));
    } 

private:
    double mean_, sigma_;
    P_uniform uniform_;
};

BZ_NAMESPACE_END

#endif // BZ_RAND_NORMAL_H

