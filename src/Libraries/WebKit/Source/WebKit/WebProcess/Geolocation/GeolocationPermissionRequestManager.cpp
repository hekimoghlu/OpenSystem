/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 14, 2024.
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
#include "GeolocationPermissionRequestManager.h"

#if ENABLE(GEOLOCATION)

#include "FrameInfoData.h"
#include "GeolocationIdentifier.h"
#include "MessageSenderInlines.h"
#include "WebFrame.h"
#include "WebPage.h"
#include "WebPageProxyMessages.h"
#include <WebCore/Document.h>
#include <WebCore/FrameLoader.h>
#include <WebCore/Geolocation.h>
#include <WebCore/LocalFrame.h>
#include <WebCore/SecurityOrigin.h>
#include <WebCore/SecurityOriginData.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {
using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(GeolocationPermissionRequestManager);

GeolocationPermissionRequestManager::GeolocationPermissionRequestManager(WebPage& page)
    : m_page(page)
{
}

GeolocationPermissionRequestManager::~GeolocationPermissionRequestManager() = default;

Ref<WebPage> GeolocationPermissionRequestManager::protectedPage() const
{
    return m_page.get();
}

void GeolocationPermissionRequestManager::startRequestForGeolocation(Geolocation& geolocation)
{
    auto* frame = geolocation.frame();

    ASSERT_WITH_MESSAGE(frame, "It is not well understood in which cases the Geolocation is alive after its frame goes away. If you hit this assertion, please add a test covering this case.");
    if (!frame) {
        geolocation.setIsAllowed(false, { });
        return;
    }

    GeolocationIdentifier geolocationID = GeolocationIdentifier::generate();

    m_geolocationToIDMap.set(geolocation, geolocationID);
    m_idToGeolocationMap.set(geolocationID, geolocation);

    auto webFrame = WebFrame::fromCoreFrame(*frame);
    ASSERT(webFrame);

    protectedPage()->send(Messages::WebPageProxy::RequestGeolocationPermissionForFrame(geolocationID, webFrame->info()));
}

void GeolocationPermissionRequestManager::revokeAuthorizationToken(const String& authorizationToken)
{
    protectedPage()->send(Messages::WebPageProxy::RevokeGeolocationAuthorizationToken(authorizationToken));
}

void GeolocationPermissionRequestManager::cancelRequestForGeolocation(Geolocation& geolocation)
{
    if (auto geolocationID = m_geolocationToIDMap.takeOptional(geolocation))
        m_idToGeolocationMap.remove(*geolocationID);
}

void GeolocationPermissionRequestManager::didReceiveGeolocationPermissionDecision(GeolocationIdentifier geolocationID, const String& authorizationToken)
{
    RefPtr geolocation = m_idToGeolocationMap.take(geolocationID).get();
    if (!geolocation)
        return;
    m_geolocationToIDMap.remove(geolocation.get());

    geolocation->setIsAllowed(!authorizationToken.isNull(), authorizationToken);
}

void GeolocationPermissionRequestManager::ref() const
{
    m_page->ref();
}

void GeolocationPermissionRequestManager::deref() const
{
    m_page->deref();
}

} // namespace WebKit

#endif // ENABLE(GEOLOCATION)
