/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 13, 2024.
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

#include "SecurityOriginData.h"
#include <wtf/Hasher.h>
#include <wtf/URL.h>

namespace WebCore {

struct ClientOrigin;
class RegistrableDomain;

class ServiceWorkerRegistrationKey {
public:
    ServiceWorkerRegistrationKey() = default;
    WEBCORE_EXPORT ServiceWorkerRegistrationKey(SecurityOriginData&& topOrigin, URL&& scope);

    WEBCORE_EXPORT static ServiceWorkerRegistrationKey emptyKey();

    friend bool operator==(const ServiceWorkerRegistrationKey&, const ServiceWorkerRegistrationKey&) = default;
    bool isEmpty() const { return *this == emptyKey(); }
    WEBCORE_EXPORT bool isMatching(const SecurityOriginData& topOrigin, const URL& clientURL) const;
    bool originIsMatching(const SecurityOriginData& topOrigin, const URL& clientURL) const;
    size_t scopeLength() const { return m_scope.string().length(); }

    WEBCORE_EXPORT ClientOrigin clientOrigin() const;
    const SecurityOriginData& topOrigin() const { return m_topOrigin; }
    WEBCORE_EXPORT RegistrableDomain firstPartyForCookies() const;
    const URL& scope() const { return m_scope; }
    void setScope(URL&& scope) { m_scope = WTFMove(scope); }

    bool relatesToOrigin(const SecurityOriginData&) const;

    WEBCORE_EXPORT ServiceWorkerRegistrationKey isolatedCopy() const &;
    WEBCORE_EXPORT ServiceWorkerRegistrationKey isolatedCopy() &&;

    String toDatabaseKey() const;
    WEBCORE_EXPORT static std::optional<ServiceWorkerRegistrationKey> fromDatabaseKey(const String&);

#if !LOG_DISABLED
    String loggingString() const;
#endif

private:
    friend struct HashTraits<ServiceWorkerRegistrationKey>;

    SecurityOriginData m_topOrigin;
    URL m_scope;
};

inline void add(Hasher& hasher, const ServiceWorkerRegistrationKey& key)
{
    add(hasher, key.topOrigin(), key.scope());
}

} // namespace WebCore

namespace WTF {

struct ServiceWorkerRegistrationKeyHash {
    static unsigned hash(const WebCore::ServiceWorkerRegistrationKey& key) { return computeHash(key); }
    static bool equal(const WebCore::ServiceWorkerRegistrationKey& a, const WebCore::ServiceWorkerRegistrationKey& b) { return a == b; }
    static const bool safeToCompareToEmptyOrDeleted = false;
};

template<> struct HashTraits<WebCore::ServiceWorkerRegistrationKey> : GenericHashTraits<WebCore::ServiceWorkerRegistrationKey> {
    static WebCore::ServiceWorkerRegistrationKey emptyValue() { return WebCore::ServiceWorkerRegistrationKey::emptyKey(); }

    static void constructDeletedValue(WebCore::ServiceWorkerRegistrationKey& slot) { new (NotNull, &slot.m_topOrigin) WebCore::SecurityOriginData(HashTableDeletedValue); }
    static bool isDeletedValue(const WebCore::ServiceWorkerRegistrationKey& slot) { return slot.m_topOrigin.isHashTableDeletedValue(); }
};

template<> struct DefaultHash<WebCore::ServiceWorkerRegistrationKey> : ServiceWorkerRegistrationKeyHash { };

} // namespace WTF
