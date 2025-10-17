/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 13, 2024.
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
#ifndef BZ_RANDOM_H
#define BZ_RANDOM_H

#ifndef BZ_BLITZ_H
 #include <blitz/blitz.h>
#endif

BZ_NAMESPACE(blitz)

template<typename P_distribution>
class Random {

public:
    typedef P_distribution T_distribution;
    typedef _bz_typename T_distribution::T_numtype T_numtype;

    Random(double parm1=0.0, double parm2=1.0, double parm3=0.0)
        : generator_(parm1, parm2, parm3)
    { }

    void randomize()
    { generator_.randomize(); }
   
    T_numtype random()
    { return generator_.random(); }

    operator T_numtype()
    { return generator_.random(); }

protected: 
    T_distribution generator_;
};

BZ_NAMESPACE_END

#include <blitz/randref.h>

#endif // BZ_RANDOM_H

