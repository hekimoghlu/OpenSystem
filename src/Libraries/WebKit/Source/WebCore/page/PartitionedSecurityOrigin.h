/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 27, 2024.
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

#include "SecurityOrigin.h"
#include <wtf/HashTraits.h>
#include <wtf/Hasher.h>
#include <wtf/Ref.h>

namespace WebCore {

struct PartitionedSecurityOrigin {
    PartitionedSecurityOrigin(Ref<SecurityOrigin>&& topOrigin, Ref<SecurityOrigin>&& clientOrigin)
        : topOrigin(WTFMove(topOrigin))
        , clientOrigin(WTFMove(clientOrigin))
    { }

    PartitionedSecurityOrigin(WTF::HashTableDeletedValueType)
        : topOrigin(WTF::HashTableDeletedValue)
        , clientOrigin(WTF::HashTableDeletedValue)
    { }

    PartitionedSecurityOrigin(WTF::HashTableEmptyValueType)
        : topOrigin(WTF::HashTableEmptyValue)
        , clientOrigin(WTF::HashTableEmptyValue)
    { }

    bool isHashTableDeletedValue() const { return topOrigin.isHashTableDeletedValue(); }
    bool isHashTableEmptyValue() const { return topOrigin.isHashTableEmptyValue(); }

    PartitionedSecurityOrigin isolatedCopy() const { return { topOrigin->isolatedCopy(), clientOrigin->isolatedCopy() }; }

    Ref<SecurityOrigin> topOrigin;
    Ref<SecurityOrigin> clientOrigin;
};

inline bool operator==(const PartitionedSecurityOrigin& a, const PartitionedSecurityOrigin& b)
{
    return a.topOrigin->isSameOriginAs(b.topOrigin) && a.clientOrigin->isSameOriginAs(b.clientOrigin);
}

} // namespace WebCore

namespace WTF {

inline void add(Hasher& hasher, const WebCore::PartitionedSecurityOrigin& origin)
{
    add(hasher, origin.topOrigin.get(), origin.clientOrigin.get());
}

struct PartitionedSecurityOriginHash {
    static unsigned hash(const WebCore::PartitionedSecurityOrigin& origin) { return computeHash(origin); }
    static bool equal(const WebCore::PartitionedSecurityOrigin& a, const WebCore::PartitionedSecurityOrigin& b) { return a == b; }
    static constexpr bool safeToCompareToEmptyOrDeleted = false;
};

template<> struct DefaultHash<WebCore::PartitionedSecurityOrigin> : PartitionedSecurityOriginHash { };

template<> struct HashTraits<WebCore::PartitionedSecurityOrigin> : SimpleClassHashTraits<WebCore::PartitionedSecurityOrigin> {
    static constexpr bool emptyValueIsZero = true;
    static WebCore::PartitionedSecurityOrigin emptyValue() { return HashTableEmptyValue; }

    template <typename>
    static void constructEmptyValue(WebCore::PartitionedSecurityOrigin& slot)
    {
        new (NotNull, std::addressof(slot)) WebCore::PartitionedSecurityOrigin(HashTableEmptyValue);
    }

    static constexpr bool hasIsEmptyValueFunction = true;
    static bool isEmptyValue(const WebCore::PartitionedSecurityOrigin& value) { return value.isHashTableEmptyValue(); }

    using PeekType = std::optional<WebCore::PartitionedSecurityOrigin>;
    static PeekType peek(const WebCore::PartitionedSecurityOrigin& value) { return isEmptyValue(value) ? std::nullopt : std::optional { value }; }

    using TakeType = std::optional<WebCore::PartitionedSecurityOrigin>;
    static TakeType take(WebCore::PartitionedSecurityOrigin&& value) { return isEmptyValue(value) ? std::nullopt : std::optional { WTFMove(value) }; }
};

} // namespace WTF
