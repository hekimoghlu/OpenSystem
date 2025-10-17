/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 4, 2024.
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
#include "GeolocationPermissionRequestManagerProxy.h"

#include "MessageSenderInlines.h"
#include "WebPageMessages.h"
#include "WebPageProxy.h"
#include "WebProcessProxy.h"
#include <wtf/UUID.h>

namespace WebKit {

GeolocationPermissionRequestManagerProxy::GeolocationPermissionRequestManagerProxy(WebPageProxy& page)
    : m_page(page)
{
}

void GeolocationPermissionRequestManagerProxy::invalidateRequests()
{
    for (auto& request : m_pendingRequests.values())
        request->invalidate();

    m_pendingRequests.clear();
}

Ref<GeolocationPermissionRequestProxy> GeolocationPermissionRequestManagerProxy::createRequest(GeolocationIdentifier geolocationID, WebProcessProxy& process)
{
    Ref request = GeolocationPermissionRequestProxy::create(*this, geolocationID, process);
    m_pendingRequests.add(geolocationID, request.ptr());
    return request;
}

void GeolocationPermissionRequestManagerProxy::didReceiveGeolocationPermissionDecision(GeolocationIdentifier geolocationID, bool allowed)
{
    if (!m_page->hasRunningProcess())
        return;

    auto it = m_pendingRequests.find(geolocationID);
    if (it == m_pendingRequests.end())
        return;

#if ENABLE(GEOLOCATION)
    String authorizationToken = allowed ? createVersion4UUIDString() : String();
    if (!authorizationToken.isNull())
        m_validAuthorizationTokens.add(authorizationToken);
    if (RefPtr process = it->value->process())
        process->send(Messages::WebPage::DidReceiveGeolocationPermissionDecision(geolocationID, authorizationToken), m_page->webPageIDInProcess(*process));
#else
    UNUSED_PARAM(allowed);
#endif

    m_pendingRequests.remove(it);
}

bool GeolocationPermissionRequestManagerProxy::isValidAuthorizationToken(const String& authorizationToken) const
{
    return !authorizationToken.isNull() && m_validAuthorizationTokens.contains(authorizationToken);
}

void GeolocationPermissionRequestManagerProxy::revokeAuthorizationToken(const String& authorizationToken)
{
    ASSERT(isValidAuthorizationToken(authorizationToken));
    if (!isValidAuthorizationToken(authorizationToken))
        return;
    m_validAuthorizationTokens.remove(authorizationToken);
}

void GeolocationPermissionRequestManagerProxy::ref() const
{
    m_page->ref();
}

void GeolocationPermissionRequestManagerProxy::deref() const
{
    m_page->deref();
}

} // namespace WebKit
