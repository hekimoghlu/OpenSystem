/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 10, 2023.
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
#ifndef BZ_EXTREMUM_H
#define BZ_EXTREMUM_H

#ifndef BZ_BLITZ_H
 #include <blitz/blitz.h>
#endif

BZ_NAMESPACE(blitz)

// The Extremum class is used for returning extreme values and their
// locations in a numeric container.  It's a simple 2-tuple, with the
// first element being the extreme value, and the send its location.
// An object of type Extremum can be automatically converted to
// the numeric type via operator T_numtype().
template<typename P_numtype, typename P_index>
class Extremum {
public:
    typedef P_numtype T_numtype;
    typedef P_index   T_index;

    Extremum(T_numtype value, T_index index)
        : value_(value), index_(index)
    { }

    T_numtype value() const
    { return value_; }

    T_index index() const
    { return index_; }

    void setValue(T_numtype value)
    { value_ = value; }

    void setIndex(T_index index)
    { index_ = index; }

    operator T_numtype() const
    { return value_; }

protected:
    T_numtype value_;
    T_index index_;
};

BZ_NAMESPACE_END

#endif // BZ_EXTREMUM_H

