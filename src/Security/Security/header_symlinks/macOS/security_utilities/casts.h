/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 15, 2024.
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
#ifndef casts_h
#define casts_h

#include <stdexcept>
#include <security_utilities/debugging.h>
#include <syslog.h>

template<typename TSource, typename TResult>
static inline TResult int_cast(TSource value) {
    // TODO: if we're using C++11, we should do some static_asserts on the signedness of these types
    TResult result = static_cast<TResult>(value);

    if (static_cast<TSource>(result) != value) {
#ifndef NDEBUG
        syslog(LOG_ERR, "%s: casted value out of range", __PRETTY_FUNCTION__);
#endif
        secnotice("int_cast", "casted value out of range");
        throw std::out_of_range("int_cast: casted value out of range");
    }
    return result;
}

#endif /* casts_h */
