/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 22, 2025.
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
#include "MediaKeySystemPermissionRequestManagerProxy.h"

#include "APISecurityOrigin.h"
#include "APIUIClient.h"
#include "Logging.h"
#include "MessageSenderInlines.h"
#include "WebAutomationSession.h"
#include "WebFrameProxy.h"
#include "WebPageMessages.h"
#include "WebPageProxy.h"
#include "WebProcess.h"
#include "WebProcessPool.h"
#include "WebProcessProxy.h"
#include "WebsiteDataStore.h"
#include <WebCore/MediaKeySystemRequest.h>
#include <WebCore/SecurityOriginData.h>
#include <wtf/Scope.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {
using namespace WebCore;

#if !RELEASE_LOG_DISABLED
static ASCIILiteral logClassName()
{
    return "MediaKeySystemPermissionRequestManagerProxy"_s;
}

static WTFLogChannel& logChannel()
{
    return JOIN_LOG_CHANNEL_WITH_PREFIX(LOG_CHANNEL_PREFIX, EME);
}

WTF_MAKE_TZONE_ALLOCATED_IMPL(MediaKeySystemPermissionRequestManagerProxy);

const Logger& MediaKeySystemPermissionRequestManagerProxy::logger() const
{
    return m_page->logger();
}
#endif

MediaKeySystemPermissionRequestManagerProxy::MediaKeySystemPermissionRequestManagerProxy(WebPageProxy& page)
    : m_page(page)
#if !RELEASE_LOG_DISABLED
    , m_logIdentifier(LoggerHelper::uniqueLogIdentifier())
#endif
{
}

MediaKeySystemPermissionRequestManagerProxy::~MediaKeySystemPermissionRequestManagerProxy()
{
    invalidatePendingRequests();
}

void MediaKeySystemPermissionRequestManagerProxy::invalidatePendingRequests()
{
    for (auto& request : m_pendingRequests.values())
        request->invalidate();

    m_pendingRequests.clear();
}

void MediaKeySystemPermissionRequestManagerProxy::denyRequest(MediaKeySystemPermissionRequestProxy& request, const String& message)
{
    if (!m_page->hasRunningProcess())
        return;

    ALWAYS_LOG(LOGIDENTIFIER, request.mediaKeySystemID().toUInt64(), ", reason: ", message);

#if ENABLE(ENCRYPTED_MEDIA)
    m_page->legacyMainFrameProcess().send(Messages::WebPage::MediaKeySystemWasDenied(request.mediaKeySystemID(), message), m_page->webPageIDInMainFrameProcess());
#else
    UNUSED_PARAM(message);
#endif
}

void MediaKeySystemPermissionRequestManagerProxy::grantRequest(MediaKeySystemPermissionRequestProxy& request)
{
    if (!m_page->hasRunningProcess())
        return;

#if ENABLE(ENCRYPTED_MEDIA)
    ALWAYS_LOG(LOGIDENTIFIER, request.mediaKeySystemID().toUInt64(), ", keySystem: ", request.keySystem());

    m_page->legacyMainFrameProcess().send(Messages::WebPage::MediaKeySystemWasGranted { request.mediaKeySystemID() }, m_page->webPageIDInMainFrameProcess());
#else
    UNUSED_PARAM(request);
#endif
}

Ref<MediaKeySystemPermissionRequestProxy> MediaKeySystemPermissionRequestManagerProxy::createRequestForFrame(MediaKeySystemRequestIdentifier mediaKeySystemID, FrameIdentifier frameID, Ref<SecurityOrigin>&& topLevelDocumentOrigin, const String& keySystem)
{
    ALWAYS_LOG(LOGIDENTIFIER, mediaKeySystemID.toUInt64());
    auto request = MediaKeySystemPermissionRequestProxy::create(*this, mediaKeySystemID, m_page->mainFrame()->frameID(), frameID, WTFMove(topLevelDocumentOrigin), keySystem);
    m_pendingRequests.add(mediaKeySystemID, request.ptr());
    return request;
}

void MediaKeySystemPermissionRequestManagerProxy::ref() const
{
    m_page->ref();
}

void MediaKeySystemPermissionRequestManagerProxy::deref() const
{
    m_page->deref();
}

} // namespace WebKit
