/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 16, 2023.
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

#include <wtf/HashTraits.h>
#include <wtf/Hasher.h>
#include <wtf/ObjectIdentifier.h>
#include <wtf/UUID.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

enum class PushSubscriptionIdentifierType { };
using PushSubscriptionIdentifier = ObjectIdentifier<PushSubscriptionIdentifierType>;

struct PushSubscriptionSetIdentifier {
    String bundleIdentifier;
    String pushPartition;
    Markable<WTF::UUID> dataStoreIdentifier;

    bool operator==(const PushSubscriptionSetIdentifier&) const;
    void add(Hasher&, const PushSubscriptionSetIdentifier&);
    bool isHashTableDeletedValue() const;

    PushSubscriptionSetIdentifier isolatedCopy() const &;
    PushSubscriptionSetIdentifier isolatedCopy() &&;

    WEBCORE_EXPORT String debugDescription() const;
};

WEBCORE_EXPORT String makePushTopic(const PushSubscriptionSetIdentifier&, const String& scope);

inline bool PushSubscriptionSetIdentifier::operator==(const PushSubscriptionSetIdentifier& other) const
{
    // Treat null and empty strings as empty strings for the purposes of hashing and comparison. The
    // reason for this is that null and empty strings are stored as empty strings in PushDatabase
    // (the columns are marked NOT NULL). We want to be able to compare instances that use null
    // strings with instances deserialized from the database that use empty strings.
    auto makeNotNull = [](const String& s) -> const String& {
        return s.isNull() ? emptyString() : s;
    };
    return makeNotNull(bundleIdentifier) == makeNotNull(other.bundleIdentifier) && makeNotNull(pushPartition) == makeNotNull(other.pushPartition) && dataStoreIdentifier == other.dataStoreIdentifier;
}

inline void add(Hasher& hasher, const PushSubscriptionSetIdentifier& sub)
{
    // Treat null and empty strings as empty strings for the purposes of hashing and comparison. See
    // the comment in operator== for more explanation.
    auto makeNotNull = [](const String& s) -> const String& {
        return s.isNull() ? emptyString() : s;
    };
    if (sub.dataStoreIdentifier)
        return add(hasher, makeNotNull(sub.bundleIdentifier), makeNotNull(sub.pushPartition), sub.dataStoreIdentifier.value());
    return add(hasher, makeNotNull(sub.bundleIdentifier), makeNotNull(sub.pushPartition));
}

inline bool PushSubscriptionSetIdentifier::isHashTableDeletedValue() const
{
    return dataStoreIdentifier && dataStoreIdentifier->isHashTableDeletedValue();
}

inline PushSubscriptionSetIdentifier PushSubscriptionSetIdentifier::isolatedCopy() const &
{
    return { bundleIdentifier.isolatedCopy(), pushPartition.isolatedCopy(), dataStoreIdentifier };
}

inline PushSubscriptionSetIdentifier PushSubscriptionSetIdentifier::isolatedCopy() &&
{
    return { WTFMove(bundleIdentifier).isolatedCopy(), WTFMove(pushPartition).isolatedCopy(), dataStoreIdentifier };
}

} // namespace WebCore

namespace WTF {

struct PushSubscriptionSetIdentifierHash {
    static unsigned hash(const WebCore::PushSubscriptionSetIdentifier& key) { return computeHash(key); }
    static bool equal(const WebCore::PushSubscriptionSetIdentifier& a, const WebCore::PushSubscriptionSetIdentifier& b) { return a == b; }
    static const bool safeToCompareToEmptyOrDeleted = true;
};

template<> struct DefaultHash<WebCore::PushSubscriptionSetIdentifier> : PushSubscriptionSetIdentifierHash { };

template<> struct HashTraits<WebCore::PushSubscriptionSetIdentifier> : GenericHashTraits<WebCore::PushSubscriptionSetIdentifier> {
    static const bool emptyValueIsZero = false;
    static WebCore::PushSubscriptionSetIdentifier emptyValue() { return { emptyString(), emptyString(), std::nullopt }; }

    static void constructDeletedValue(WebCore::PushSubscriptionSetIdentifier& slot) { new (NotNull, &slot) WebCore::PushSubscriptionSetIdentifier { emptyString(), emptyString(), WTF::UUID { HashTableDeletedValue } }; }
    static bool isDeletedValue(const WebCore::PushSubscriptionSetIdentifier& value) { return value.isHashTableDeletedValue(); }
};

} // namespace WTF
