/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 9, 2025.
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
#include <wtf/text/WTFString.h>

namespace WebCore {

struct CookieChangeSubscription {
    String name;
    String url;

    CookieChangeSubscription() = default;

    CookieChangeSubscription(String&& name, String&& url)
        : name(WTFMove(name))
        , url(WTFMove(url))
    { }

    CookieChangeSubscription isolatedCopy() const & { return { name.isolatedCopy(), url.isolatedCopy() }; }
    CookieChangeSubscription isolatedCopy() && { return { WTFMove(name).isolatedCopy(), WTFMove(url).isolatedCopy() }; }

    explicit CookieChangeSubscription(WTF::HashTableDeletedValueType deletedValue)
        : name(deletedValue)
    { }

    bool operator==(const CookieChangeSubscription& other) const = default;

    bool isHashTableDeletedValue() const { return name.isHashTableDeletedValue(); }
};

} // namespace WebCore

namespace WTF {

struct CookieChangeSubscriptionHash {
    static unsigned hash(const WebCore::CookieChangeSubscription& subscription)
    {
        return computeHash(subscription.name, subscription.url);
    }

    static bool equal(const WebCore::CookieChangeSubscription& a, const WebCore::CookieChangeSubscription& b)
    {
        return a == b;
    }

    static const bool safeToCompareToEmptyOrDeleted = false;
};

template<typename T> struct DefaultHash;
template<> struct DefaultHash<WebCore::CookieChangeSubscription> : CookieChangeSubscriptionHash { };

template<> struct HashTraits<WebCore::CookieChangeSubscription> : SimpleClassHashTraits<WebCore::CookieChangeSubscription> {
    static const bool emptyValueIsZero = false;
    static const bool hasIsEmptyValueFunction = false;
};

} // namespace WTF
