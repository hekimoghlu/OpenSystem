/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 4, 2025.
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

#include "ProtectionSpace.h"
#include <wtf/HashTraits.h>
#include <wtf/Hasher.h>

namespace WebCore {

struct ProtectionSpaceHash {
    static unsigned hash(const ProtectionSpace& protectionSpace)
    { 
        Hasher hasher;
        add(hasher, protectionSpace.host());
        add(hasher, protectionSpace.port());
        add(hasher, protectionSpace.serverType());
        add(hasher, protectionSpace.authenticationScheme());
        if (!protectionSpace.isProxy())
            add(hasher, protectionSpace.realm());
        return hasher.hash();
    }
    
    static bool equal(const ProtectionSpace& a, const ProtectionSpace& b) { return a == b; }
    static constexpr bool safeToCompareToEmptyOrDeleted = false;
};

} // namespace WebCore

namespace WTF {

template<> struct HashTraits<WebCore::ProtectionSpace> : SimpleClassHashTraits<WebCore::ProtectionSpace> {
    static constexpr bool emptyValueIsZero = false;
};
template<> struct DefaultHash<WebCore::ProtectionSpace> : WebCore::ProtectionSpaceHash { };

} // namespace WTF
