/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 24, 2022.
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
#include "WebGeolocationManagerProxy.h"

#include "WebGeolocationManagerMessages.h"
#include "WebProcessProxy.h"

namespace WebKit {

// FIXME: This should be used by all Cocoa ports
#if PLATFORM(IOS_FAMILY)

void WebGeolocationManagerProxy::positionChanged(const String& websiteIdentifier, WebCore::GeolocationPositionData&& position)
{
    auto registrableDomain = WebCore::RegistrableDomain::uncheckedCreateFromRegistrableDomainString(websiteIdentifier);
    auto it = m_perDomainData.find(registrableDomain);
    if (it == m_perDomainData.end())
        return;

    auto& perDomainData = *it->value;
    perDomainData.lastPosition = WTFMove(position);
    for (auto& webProcessProxy : perDomainData.watchers)
        webProcessProxy.send(Messages::WebGeolocationManager::DidChangePosition(registrableDomain, perDomainData.lastPosition.value()), 0);
}

void WebGeolocationManagerProxy::errorOccurred(const String& websiteIdentifier, const String& errorMessage)
{
    auto registrableDomain = WebCore::RegistrableDomain::uncheckedCreateFromRegistrableDomainString(websiteIdentifier);
    auto it = m_perDomainData.find(registrableDomain);
    if (it == m_perDomainData.end())
        return;

    auto& perDomainData = *it->value;
    for (auto& webProcessProxy : perDomainData.watchers)
        webProcessProxy.send(Messages::WebGeolocationManager::DidFailToDeterminePosition(registrableDomain, errorMessage), 0);
}

void WebGeolocationManagerProxy::resetGeolocation(const String& websiteIdentifier)
{
    auto registrableDomain = WebCore::RegistrableDomain::uncheckedCreateFromRegistrableDomainString(websiteIdentifier);
    auto it = m_perDomainData.find(registrableDomain);
    if (it == m_perDomainData.end())
        return;

    auto& perDomainData = *it->value;
    for (auto& webProcessProxy : perDomainData.watchers)
        webProcessProxy.send(Messages::WebGeolocationManager::ResetPermissions(registrableDomain), 0);
}

#endif

} // namespace WebKit
