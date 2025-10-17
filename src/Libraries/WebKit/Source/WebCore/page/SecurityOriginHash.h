/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 29, 2022.
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

#include <wtf/Hasher.h>
#include <wtf/URL.h>
#include "SecurityOrigin.h"
#include <wtf/RefPtr.h>

namespace WebCore {

struct SecurityOriginHash {
    static unsigned hash(const SecurityOrigin* origin)
    {
        return computeHash(*origin);
    }
    static unsigned hash(const RefPtr<SecurityOrigin>& origin)
    {
        return hash(origin.get());
    }

    static bool equal(const SecurityOrigin* a, const SecurityOrigin* b)
    {
        if (!a || !b)
            return a == b;
        return a->isSameSchemeHostPort(*b);
    }
    static bool equal(const SecurityOrigin* a, const RefPtr<SecurityOrigin>& b)
    {
        return equal(a, b.get());
    }
    static bool equal(const RefPtr<SecurityOrigin>& a, const SecurityOrigin* b)
    {
        return equal(a.get(), b);
    }
    static bool equal(const RefPtr<SecurityOrigin>& a, const RefPtr<SecurityOrigin>& b)
    {
        return equal(a.get(), b.get());
    }

    static const bool safeToCompareToEmptyOrDeleted = false;
};

} // namespace WebCore

namespace WTF {

template<typename> struct DefaultHash;
template<> struct DefaultHash<RefPtr<WebCore::SecurityOrigin>> : WebCore::SecurityOriginHash { };

} // namespace WTF
