/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 11, 2021.
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
#include "config.h"
#include "ServiceWorkerRegistrationKey.h"

#include "ClientOrigin.h"
#include "RegistrableDomain.h"
#include "SecurityOrigin.h"
#include <wtf/URLHash.h>
#include <wtf/text/MakeString.h>
#include <wtf/text/StringToIntegerConversion.h>

namespace WebCore {

ServiceWorkerRegistrationKey::ServiceWorkerRegistrationKey(SecurityOriginData&& topOrigin, URL&& scope)
    : m_topOrigin(WTFMove(topOrigin))
    , m_scope(WTFMove(scope))
{
    ASSERT(!m_scope.hasFragmentIdentifier());
}

ServiceWorkerRegistrationKey ServiceWorkerRegistrationKey::emptyKey()
{
    return { };
}

ServiceWorkerRegistrationKey ServiceWorkerRegistrationKey::isolatedCopy() const &
{
    return { m_topOrigin.isolatedCopy(), m_scope.isolatedCopy() };
}

ServiceWorkerRegistrationKey ServiceWorkerRegistrationKey::isolatedCopy() &&
{
    return { WTFMove(m_topOrigin).isolatedCopy(), WTFMove(m_scope).isolatedCopy() };
}

bool ServiceWorkerRegistrationKey::isMatching(const SecurityOriginData& topOrigin, const URL& clientURL) const
{
    return originIsMatching(topOrigin, clientURL) && clientURL.string().startsWith(m_scope.string());
}

bool ServiceWorkerRegistrationKey::originIsMatching(const SecurityOriginData& topOrigin, const URL& clientURL) const
{
    if (topOrigin != m_topOrigin)
        return false;

    return protocolHostAndPortAreEqual(clientURL, m_scope);
}

bool ServiceWorkerRegistrationKey::relatesToOrigin(const SecurityOriginData& securityOrigin) const
{
    if (m_topOrigin == securityOrigin)
        return true;

    return SecurityOriginData::fromURL(m_scope) == securityOrigin;
}

RegistrableDomain ServiceWorkerRegistrationKey::firstPartyForCookies() const
{
    return RegistrableDomain::uncheckedCreateFromHost(m_topOrigin.host());
}

static const char separatorCharacter = '_';

String ServiceWorkerRegistrationKey::toDatabaseKey() const
{
    if (m_topOrigin.port())
        return makeString(m_topOrigin.protocol(), separatorCharacter, m_topOrigin.host(), separatorCharacter, String::number(m_topOrigin.port().value()), separatorCharacter, m_scope.string());
    return makeString(m_topOrigin.protocol(), separatorCharacter, m_topOrigin.host(), separatorCharacter, separatorCharacter, m_scope.string());
}

std::optional<ServiceWorkerRegistrationKey> ServiceWorkerRegistrationKey::fromDatabaseKey(const String& key)
{
    auto first = key.find(separatorCharacter, 0);
    if (first == notFound)
        return std::nullopt;

    auto second = key.find(separatorCharacter, first + 1);
    if (second == notFound)
        return std::nullopt;

    auto third = key.find(separatorCharacter, second + 1);
    if (third == notFound)
        return std::nullopt;

    std::optional<uint16_t> shortPort;

    // If there's a gap between third and second, we expect to have a port to decode
    if (third - second > 1) {
        shortPort = parseInteger<uint16_t>(StringView { key }.substring(second + 1, third - second - 1));
        if (!shortPort)
            return std::nullopt;
    }

    auto scheme = StringView(key).left(first);
    auto host = StringView(key).substring(first + 1, second - first - 1);

    URL topOriginURL { makeString(scheme, "://"_s, host) };
    if (!topOriginURL.isValid())
        return std::nullopt;

    URL scope { key.substring(third + 1) };
    if (!scope.isValid())
        return std::nullopt;

    SecurityOriginData topOrigin { scheme.toString(), host.toString(), shortPort };
    return ServiceWorkerRegistrationKey { WTFMove(topOrigin), WTFMove(scope) };
}

ClientOrigin ServiceWorkerRegistrationKey::clientOrigin() const
{
    return ClientOrigin { m_topOrigin, SecurityOriginData::fromURL(m_scope) };
}

#if !LOG_DISABLED
String ServiceWorkerRegistrationKey::loggingString() const
{
    return makeString(m_topOrigin.debugString(), '-', m_scope.string());
}
#endif

} // namespace WebCore
