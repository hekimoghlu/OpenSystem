/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 25, 2022.
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
#include "WebUserMediaClient.h"

#if ENABLE(MEDIA_STREAM)

#include "MessageSenderInlines.h"
#include "UserMediaPermissionRequestManager.h"
#include "WebPage.h"
#include "WebPageProxyMessages.h"
#include <WebCore/UserMediaController.h>
#include <WebCore/UserMediaRequest.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {
using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebUserMediaClient);

WebUserMediaClient::WebUserMediaClient(WebPage& page)
    : m_page(page)
{
}

Ref<WebPage> WebUserMediaClient::protectedPage() const
{
    return m_page.get();
}

void WebUserMediaClient::pageDestroyed()
{
    delete this;
}

void WebUserMediaClient::requestUserMediaAccess(UserMediaRequest& request)
{
    protectedPage()->protectedUserMediaPermissionRequestManager()->startUserMediaRequest(request);
}

void WebUserMediaClient::cancelUserMediaAccessRequest(UserMediaRequest& request)
{
    protectedPage()->protectedUserMediaPermissionRequestManager()->cancelUserMediaRequest(request);
}

void WebUserMediaClient::enumerateMediaDevices(Document& document, UserMediaClient::EnumerateDevicesCallback&& completionHandler)
{
    protectedPage()->protectedUserMediaPermissionRequestManager()->enumerateMediaDevices(document, WTFMove(completionHandler));
}

WebUserMediaClient::DeviceChangeObserverToken WebUserMediaClient::addDeviceChangeObserver(WTF::Function<void()>&& observer)
{
    return protectedPage()->protectedUserMediaPermissionRequestManager()->addDeviceChangeObserver(WTFMove(observer));
}

void WebUserMediaClient::removeDeviceChangeObserver(DeviceChangeObserverToken token)
{
    protectedPage()->protectedUserMediaPermissionRequestManager()->removeDeviceChangeObserver(token);
}

void WebUserMediaClient::updateCaptureState(const WebCore::Document& document, bool isActive, WebCore::MediaProducerMediaCaptureKind kind, CompletionHandler<void(std::optional<WebCore::Exception>&&)>&& completionHandler)
{
    protectedPage()->protectedUserMediaPermissionRequestManager()->updateCaptureState(document, isActive, kind, WTFMove(completionHandler));
}

void WebUserMediaClient::setShouldListenToVoiceActivity(bool shouldListen)
{
    protectedPage()->send(Messages::WebPageProxy::SetShouldListenToVoiceActivity { shouldListen });
}

} // namespace WebKit;

#endif // MEDIA_STREAM
