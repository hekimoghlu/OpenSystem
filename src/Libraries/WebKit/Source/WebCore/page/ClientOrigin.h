/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 23, 2024.
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

#include "RegistrableDomain.h"
#include "SecurityOriginData.h"
#include <wtf/HashTraits.h>
#include <wtf/Hasher.h>
#include <wtf/URL.h>
#include <wtf/text/MakeString.h>

namespace WebCore {

struct ClientOrigin {
    static ClientOrigin emptyKey() { return { }; }

    friend bool operator==(const ClientOrigin&, const ClientOrigin&) = default;

    ClientOrigin isolatedCopy() const & { return { topOrigin.isolatedCopy(), clientOrigin.isolatedCopy() }; }
    ClientOrigin isolatedCopy() && { return { WTFMove(topOrigin).isolatedCopy(), WTFMove(clientOrigin).isolatedCopy() }; }
    bool isRelated(const SecurityOriginData& other) const { return topOrigin == other || clientOrigin == other; }

    RegistrableDomain clientRegistrableDomain() const { return RegistrableDomain::uncheckedCreateFromHost(clientOrigin.host()); }

    SecurityOriginData topOrigin;
    SecurityOriginData clientOrigin;

    String loggingString() const { return makeString(topOrigin.toString(), '-', clientOrigin.toString()); }
};

inline void add(Hasher& hasher, const ClientOrigin& origin)
{
    add(hasher, origin.topOrigin, origin.clientOrigin);
}

} // namespace WebCore

namespace WTF {

struct ClientOriginKeyHash {
    static unsigned hash(const WebCore::ClientOrigin& key) { return computeHash(key); }
    static bool equal(const WebCore::ClientOrigin& a, const WebCore::ClientOrigin& b) { return a == b; }
    static const bool safeToCompareToEmptyOrDeleted = false;
};

template<> struct HashTraits<WebCore::ClientOrigin> : GenericHashTraits<WebCore::ClientOrigin> {
    static WebCore::ClientOrigin emptyValue() { return WebCore::ClientOrigin::emptyKey(); }

    static void constructDeletedValue(WebCore::ClientOrigin& slot) { new (NotNull, &slot.topOrigin) WebCore::SecurityOriginData(WTF::HashTableDeletedValue); }
    static bool isDeletedValue(const WebCore::ClientOrigin& slot) { return slot.topOrigin.isHashTableDeletedValue(); }
};

template<> struct DefaultHash<WebCore::ClientOrigin> : ClientOriginKeyHash { };

} // namespace WTF
