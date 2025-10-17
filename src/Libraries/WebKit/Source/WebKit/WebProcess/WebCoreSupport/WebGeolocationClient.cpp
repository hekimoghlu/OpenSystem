/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 8, 2024.
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
#include "WebGeolocationClient.h"

#if ENABLE(GEOLOCATION)

#include "GeolocationPermissionRequestManager.h"
#include "WebGeolocationManager.h"
#include "WebProcess.h"
#include <WebCore/Geolocation.h>
#include <WebCore/GeolocationPositionData.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {
using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebGeolocationClient);

WebGeolocationClient::~WebGeolocationClient()
{
}

void WebGeolocationClient::geolocationDestroyed()
{
    WebProcess::singleton().supplement<WebGeolocationManager>()->unregisterWebPage(m_page.get());
    delete this;
}

void WebGeolocationClient::startUpdating(const String& authorizationToken, bool needsHighAccuracy)
{
    WebProcess::singleton().supplement<WebGeolocationManager>()->registerWebPage(m_page.get(), authorizationToken, needsHighAccuracy);
}

void WebGeolocationClient::stopUpdating()
{
    WebProcess::singleton().supplement<WebGeolocationManager>()->unregisterWebPage(m_page.get());
}

void WebGeolocationClient::setEnableHighAccuracy(bool enabled)
{
    WebProcess::singleton().supplement<WebGeolocationManager>()->setEnableHighAccuracyForPage(m_page.get(), enabled);
}

std::optional<GeolocationPositionData> WebGeolocationClient::lastPosition()
{
    return std::nullopt;
}

void WebGeolocationClient::requestPermission(Geolocation& geolocation)
{
    m_page.get().geolocationPermissionRequestManager().startRequestForGeolocation(geolocation);
}

void WebGeolocationClient::revokeAuthorizationToken(const String& authorizationToken)
{
    m_page.get().geolocationPermissionRequestManager().revokeAuthorizationToken(authorizationToken);
}

void WebGeolocationClient::cancelPermissionRequest(Geolocation& geolocation)
{
    m_page.get().geolocationPermissionRequestManager().cancelRequestForGeolocation(geolocation);
}

} // namespace WebKit

#endif // ENABLE(GEOLOCATION)
