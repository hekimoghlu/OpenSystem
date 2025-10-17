/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 11, 2021.
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
#ifndef URLHash_h
#define URLHash_h

#include <wtf/URL.h>
#include <wtf/text/StringHash.h>
#include <wtf/text/WTFString.h>

namespace WTF {

struct URLHash {
    static unsigned hash(const URL& key)
    {
        return key.string().impl()->hash();
    }

    static bool equal(const URL& a, const URL& b)
    {
        return StringHash::equal(a.string(), b.string());
    }

    static constexpr bool safeToCompareToEmptyOrDeleted = false;
    static constexpr bool hasHashInValue = true;
};

// URLHash is the default hash for String
template<> struct DefaultHash<URL> : URLHash { };

template<> struct HashTraits<URL> : SimpleClassHashTraits<URL> { };

} // namespace WTF

#endif // URLHash_h
