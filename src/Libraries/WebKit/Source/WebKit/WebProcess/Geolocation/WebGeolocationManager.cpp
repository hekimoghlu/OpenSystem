/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 31, 2023.
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
#include "WebGeolocationManager.h"

#include "WebGeolocationManagerMessages.h"
#include "WebGeolocationManagerProxyMessages.h"
#include "WebPage.h"
#include "WebProcess.h"
#include <WebCore/Geolocation.h>
#include <WebCore/GeolocationController.h>
#include <WebCore/GeolocationError.h>
#include <WebCore/GeolocationPositionData.h>
#include <WebCore/LocalFrame.h>
#include <WebCore/Page.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {
using namespace WebCore;

static RegistrableDomain registrableDomainForPage(WebPage& page)
{
    RefPtr corePage = page.protectedCorePage();
    if (!corePage)
        return { };

    return RegistrableDomain { corePage->mainFrameURL() };
}

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebGeolocationManager);

ASCIILiteral WebGeolocationManager::supplementName()
{
    return "WebGeolocationManager"_s;
}

WebGeolocationManager::WebGeolocationManager(WebProcess& process)
    : m_process(process)
{
    process.addMessageReceiver(Messages::WebGeolocationManager::messageReceiverName(), *this);
}

WebGeolocationManager::~WebGeolocationManager() = default;

void WebGeolocationManager::ref() const
{
    m_process->ref();
}

void WebGeolocationManager::deref() const
{
    m_process->deref();
}

void WebGeolocationManager::registerWebPage(WebPage& page, const String& authorizationToken, bool needsHighAccuracy)
{
    auto registrableDomain = registrableDomainForPage(page);
    if (registrableDomain.string().isEmpty())
        return;

    auto& pageSets = m_pageSets.add(registrableDomain, PageSets()).iterator->value;
    bool wasUpdating = isUpdating(pageSets);
    bool highAccuracyWasEnabled = isHighAccuracyEnabled(pageSets);

    pageSets.pageSet.add(page);
    if (needsHighAccuracy)
        pageSets.highAccuracyPageSet.add(page);
    m_pageToRegistrableDomain.add(page, registrableDomain);

    if (!wasUpdating) {
        WebProcess::singleton().parentProcessConnection()->send(Messages::WebGeolocationManagerProxy::StartUpdating(registrableDomain, page.webPageProxyIdentifier(), authorizationToken, needsHighAccuracy), 0);
        return;
    }

    bool highAccuracyShouldBeEnabled = isHighAccuracyEnabled(pageSets);
    if (highAccuracyWasEnabled != highAccuracyShouldBeEnabled)
        WebProcess::singleton().parentProcessConnection()->send(Messages::WebGeolocationManagerProxy::SetEnableHighAccuracy(registrableDomain, highAccuracyShouldBeEnabled), 0);
}

void WebGeolocationManager::unregisterWebPage(WebPage& page)
{
    auto registrableDomain = m_pageToRegistrableDomain.take(page);
    if (registrableDomain.string().isEmpty())
        return;

    auto it = m_pageSets.find(registrableDomain);
    if (it == m_pageSets.end())
        return;

    auto& pageSets = it->value;
    bool highAccuracyWasEnabled = isHighAccuracyEnabled(pageSets);

    pageSets.pageSet.remove(page);
    pageSets.highAccuracyPageSet.remove(page);

    if (!isUpdating(pageSets))
        WebProcess::singleton().parentProcessConnection()->send(Messages::WebGeolocationManagerProxy::StopUpdating(registrableDomain), 0);
    else {
        bool highAccuracyShouldBeEnabled = isHighAccuracyEnabled(pageSets);
        if (highAccuracyWasEnabled != highAccuracyShouldBeEnabled)
            WebProcess::singleton().parentProcessConnection()->send(Messages::WebGeolocationManagerProxy::SetEnableHighAccuracy(registrableDomain, highAccuracyShouldBeEnabled), 0);
    }

    if (pageSets.pageSet.isEmptyIgnoringNullReferences() && pageSets.highAccuracyPageSet.isEmptyIgnoringNullReferences())
        m_pageSets.remove(it);
}

void WebGeolocationManager::setEnableHighAccuracyForPage(WebPage& page, bool enabled)
{
    auto registrableDomain = m_pageToRegistrableDomain.get(page);
    if (registrableDomain.string().isEmpty())
        return;

    auto it = m_pageSets.find(registrableDomain);
    ASSERT(it != m_pageSets.end());
    if (it == m_pageSets.end())
        return;

    auto& pageSets = it->value;
    bool highAccuracyWasEnabled = isHighAccuracyEnabled(pageSets);

    if (enabled)
        pageSets.highAccuracyPageSet.add(page);
    else
        pageSets.highAccuracyPageSet.remove(page);

    bool highAccuracyShouldBeEnabled = isHighAccuracyEnabled(pageSets);
    if (highAccuracyWasEnabled != isHighAccuracyEnabled(pageSets))
        WebProcess::singleton().parentProcessConnection()->send(Messages::WebGeolocationManagerProxy::SetEnableHighAccuracy(registrableDomain, highAccuracyShouldBeEnabled), 0);
}

void WebGeolocationManager::didChangePosition(const WebCore::RegistrableDomain& registrableDomain, const GeolocationPositionData& position)
{
#if ENABLE(GEOLOCATION)
    if (auto it = m_pageSets.find(registrableDomain); it != m_pageSets.end()) {
        for (auto& page : copyToVector(it->value.pageSet)) {
            if (page->corePage())
                GeolocationController::from(page->corePage())->positionChanged(position);
        }
    }
#else
    UNUSED_PARAM(registrableDomain);
    UNUSED_PARAM(position);
#endif // ENABLE(GEOLOCATION)
}

void WebGeolocationManager::didFailToDeterminePosition(const WebCore::RegistrableDomain& registrableDomain, const String& errorMessage)
{
#if ENABLE(GEOLOCATION)
    if (auto it = m_pageSets.find(registrableDomain); it != m_pageSets.end()) {
        // FIXME: Add localized error string.
        auto error = GeolocationError::create(GeolocationError::PositionUnavailable, errorMessage);

        for (auto& page : copyToVector(it->value.pageSet)) {
            if (page->corePage())
                GeolocationController::from(page->corePage())->errorOccurred(error.get());
        }
    }
#else
    UNUSED_PARAM(registrableDomain);
    UNUSED_PARAM(errorMessage);
#endif // ENABLE(GEOLOCATION)
}

bool WebGeolocationManager::isUpdating(const PageSets& pageSets) const
{
    return !pageSets.pageSet.isEmptyIgnoringNullReferences();
}

bool WebGeolocationManager::isHighAccuracyEnabled(const PageSets& pageSets) const
{
    return !pageSets.highAccuracyPageSet.isEmptyIgnoringNullReferences();
}

#if PLATFORM(IOS_FAMILY)
void WebGeolocationManager::resetPermissions(const WebCore::RegistrableDomain& registrableDomain)
{
    auto it = m_pageSets.find(registrableDomain);
    if (it != m_pageSets.end())
        return;

    for (auto& page : copyToVector(it->value.pageSet)) {
        if (RefPtr mainFrame = page->localMainFrame())
            mainFrame->resetAllGeolocationPermission();
    }
}
#endif // PLATFORM(IOS_FAMILY)

} // namespace WebKit
