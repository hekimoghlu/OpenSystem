/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 9, 2022.
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
#include "MediaKeySystemPermissionRequestManager.h"

#if ENABLE(ENCRYPTED_MEDIA)

#include "Logging.h"
#include "MessageSenderInlines.h"
#include "WebFrame.h"
#include "WebPage.h"
#include "WebPageProxyMessages.h"
#include <WebCore/Document.h>
#include <WebCore/FrameDestructionObserverInlines.h>
#include <WebCore/FrameLoader.h>
#include <WebCore/LocalFrame.h>
#include <WebCore/Page.h>
#include <WebCore/SecurityOrigin.h>
#include <WebCore/SecurityOriginData.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {
using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(MediaKeySystemPermissionRequestManager);

MediaKeySystemPermissionRequestManager::MediaKeySystemPermissionRequestManager(WebPage& page)
    : m_page(page)
{
}

void MediaKeySystemPermissionRequestManager::startMediaKeySystemRequest(MediaKeySystemRequest& request)
{
    Document* document = request.document();
    auto* frame = document ? document->frame() : nullptr;

    if (!frame || !document->page()) {
        request.deny(emptyString());
        return;
    }

    if (document->page()->canStartMedia()) {
        sendMediaKeySystemRequest(request);
        return;
    }

    auto& pendingRequests = m_pendingMediaKeySystemRequests.add(document, Vector<Ref<MediaKeySystemRequest>>()).iterator->value;
    if (pendingRequests.isEmpty())
        document->addMediaCanStartListener(*this);
    pendingRequests.append(request);
}

void MediaKeySystemPermissionRequestManager::sendMediaKeySystemRequest(MediaKeySystemRequest& userRequest)
{
    auto* frame = userRequest.document() ? userRequest.document()->frame() : nullptr;
    if (!frame) {
        userRequest.deny(emptyString());
        return;
    }

    m_ongoingMediaKeySystemRequests.add(userRequest.identifier(), userRequest);

    auto webFrame = WebFrame::fromCoreFrame(*frame);
    ASSERT(webFrame);

    auto* topLevelDocumentOrigin = userRequest.topLevelDocumentOrigin();
    Ref { m_page.get() }->send(Messages::WebPageProxy::RequestMediaKeySystemPermissionForFrame(userRequest.identifier(), webFrame->frameID(), topLevelDocumentOrigin->data(), userRequest.keySystem()));
}

void MediaKeySystemPermissionRequestManager::cancelMediaKeySystemRequest(MediaKeySystemRequest& request)
{
    if (auto removedRequest = m_ongoingMediaKeySystemRequests.take(request.identifier()))
        return;

    auto* document = request.document();
    if (!document)
        return;

    auto iterator = m_pendingMediaKeySystemRequests.find(document);
    if (iterator == m_pendingMediaKeySystemRequests.end())
        return;

    auto& pendingRequests = iterator->value;
    pendingRequests.removeFirstMatching([&request](auto& item) {
        return &request == item.ptr();
    });

    if (!pendingRequests.isEmpty())
        return;

    document->removeMediaCanStartListener(*this);
    m_pendingMediaKeySystemRequests.remove(iterator);
}

void MediaKeySystemPermissionRequestManager::mediaCanStart(Document& document)
{
    ASSERT(document.page()->canStartMedia());

    auto pendingRequests = m_pendingMediaKeySystemRequests.take(&document);
    for (auto& pendingRequest : pendingRequests)
        sendMediaKeySystemRequest(pendingRequest);
}

void MediaKeySystemPermissionRequestManager::mediaKeySystemWasGranted(MediaKeySystemRequestIdentifier requestID)
{
    auto request = m_ongoingMediaKeySystemRequests.take(requestID);
    if (!request)
        return;

    request->allow();
}

void MediaKeySystemPermissionRequestManager::mediaKeySystemWasDenied(MediaKeySystemRequestIdentifier requestID, String&& message)
{
    auto request = m_ongoingMediaKeySystemRequests.take(requestID);
    if (!request)
        return;

    request->deny(WTFMove(message));
}

void MediaKeySystemPermissionRequestManager::ref() const
{
    m_page->ref();
}

void MediaKeySystemPermissionRequestManager::deref() const
{
    m_page->deref();
}

} // namespace WebKit

#endif // ENABLE(ENCRYPTED_MEDIA)
