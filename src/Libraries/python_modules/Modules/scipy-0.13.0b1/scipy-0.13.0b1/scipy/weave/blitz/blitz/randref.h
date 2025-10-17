/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 25, 2025.
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
#ifndef BZ_RANDREF_H
#define BZ_RANDREF_H

#ifndef BZ_RANDOM_H
 #error <blitz/randref.h> must be included via <blitz/random.h>
#endif // BZ_RANDOM_H

BZ_NAMESPACE(blitz)

template<typename P_distribution>
class _bz_VecExprRandom {

public:
    typedef _bz_typename Random<P_distribution>::T_numtype T_numtype;

    _bz_VecExprRandom(Random<P_distribution>& random)
        : random_(random)
    { }

#ifdef BZ_MANUAL_VECEXPR_COPY_CONSTRUCTOR
    _bz_VecExprRandom(_bz_VecExprRandom<P_distribution>& x)
        : random_(x.random_)
    { }
#endif

    T_numtype operator[](unsigned) const
    { return random_.random(); }

    T_numtype operator()(unsigned) const
    { return random_.random(); }

    unsigned length(unsigned recommendedLength) const
    { return recommendedLength; }

    unsigned _bz_suggestLength() const
    { return 0; }

    bool _bz_hasFastAccess() const
    { return 1; }

    T_numtype _bz_fastAccess(unsigned) const
    { return random_.random(); }

private:
    _bz_VecExprRandom() : random_( Random<P_distribution>() ) { }

    Random<P_distribution>& random_;
};

BZ_NAMESPACE_END

#endif // BZ_RANDREF_H

