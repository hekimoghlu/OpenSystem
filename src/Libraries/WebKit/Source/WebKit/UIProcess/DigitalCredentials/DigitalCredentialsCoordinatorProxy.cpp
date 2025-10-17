/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 9, 2023.
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
#include "DigitalCredentialsCoordinatorProxy.h"

#if ENABLE(WEB_AUTHN)

#include "DigitalCredentialsCoordinatorProxyMessages.h"
#include "WebPageProxy.h"
#include "WebProcessProxy.h"
#include <WebCore/DigitalCredentialRequestOptions.h>
#include <WebCore/ExceptionData.h>
#include <wtf/CompletionHandler.h>

namespace WebKit {
using namespace WebCore;

Ref<DigitalCredentialsCoordinatorProxy> DigitalCredentialsCoordinatorProxy::create(WebPageProxy& page)
{
    return adoptRef(*new DigitalCredentialsCoordinatorProxy(page));
}

DigitalCredentialsCoordinatorProxy::DigitalCredentialsCoordinatorProxy(WebPageProxy& page)
    : m_page(page)
{
    page.protectedLegacyMainFrameProcess()->addMessageReceiver(Messages::DigitalCredentialsCoordinatorProxy::messageReceiverName(), page.webPageIDInMainFrameProcess(), *this);
}

DigitalCredentialsCoordinatorProxy::~DigitalCredentialsCoordinatorProxy()
{
    if (RefPtr page = m_page.get())
        page->protectedLegacyMainFrameProcess()->removeMessageReceiver(Messages::DigitalCredentialsCoordinatorProxy::messageReceiverName(), page->webPageIDInMainFrameProcess());
}

std::optional<SharedPreferencesForWebProcess> DigitalCredentialsCoordinatorProxy::sharedPreferencesForWebProcess() const
{
    if (RefPtr page = m_page.get())
        return page->protectedLegacyMainFrameProcess()->sharedPreferencesForWebProcess();
    return std::nullopt;
}

void DigitalCredentialsCoordinatorProxy::requestDigitalCredential(FrameIdentifier frameId, FrameInfoData&& frameInfo, DigitalCredentialRequestOptions&& options, DigitalRequestCompletionHandler&& handler)
{
    // FIXME: Handle the request for a digital credential.
    // For now, we will simply call the handler with a dummy exception data.
    handler({ ExceptionCode::NotSupportedError, "Not implemented"_s });
}

void DigitalCredentialsCoordinatorProxy::cancel(CompletionHandler<void()>&& completionHandler)
{
    // FIXME: Handle cancellation properly.
    completionHandler();
}

} // namespace WebKit

#endif // ENABLE(WEB_AUTHN)
