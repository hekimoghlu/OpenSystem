/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 24, 2025.
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

#include "ClientOrigin.h"

namespace WebCore {

struct SharedWorkerKey {
    ClientOrigin origin;
    URL url;
    String name;

    friend bool operator==(const SharedWorkerKey&, const SharedWorkerKey&) = default;
};

inline void add(Hasher& hasher, const SharedWorkerKey& key)
{
    add(hasher, key.origin, key.url, key.name);
}

} // namespace WebCore

namespace WTF {

template<> struct DefaultHash<WebCore::SharedWorkerKey> {
    static unsigned hash(const WebCore::SharedWorkerKey& key) { return computeHash(key); }
    static bool equal(const WebCore::SharedWorkerKey& a, const WebCore::SharedWorkerKey& b) { return a == b; }
    static const bool safeToCompareToEmptyOrDeleted = false;
};

template<> struct HashTraits<WebCore::SharedWorkerKey> : GenericHashTraits<WebCore::SharedWorkerKey> {
    static constexpr bool emptyValueIsZero = false;
    static void constructDeletedValue(WebCore::SharedWorkerKey& slot) { new (NotNull, &slot.url) URL(WTF::HashTableDeletedValue); }
    static bool isDeletedValue(const WebCore::SharedWorkerKey& slot) { return slot.url.isHashTableDeletedValue(); }
};

} // namespace WTF
