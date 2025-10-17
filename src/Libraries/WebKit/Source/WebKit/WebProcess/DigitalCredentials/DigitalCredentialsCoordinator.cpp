/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 12, 2023.
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
#include "DigitalCredentialsCoordinator.h"

#if ENABLE(WEB_AUTHN)
#include "DigitalCredentialsCoordinatorProxyMessages.h"
#include "FrameInfoData.h"
#include "WebFrame.h"
#include "WebPage.h"
#include <WebCore/DigitalCredentialRequestOptions.h>
#include <WebCore/LocalFrame.h>
#include <wtf/Logger.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {
using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(DigitalCredentialsCoordinator);

DigitalCredentialsCoordinator::DigitalCredentialsCoordinator(WebPage& page)
    : m_page(page)
{
}

RefPtr<WebPage> DigitalCredentialsCoordinator::protectedPage() const
{
    return m_page.get();
}

void DigitalCredentialsCoordinator::requestDigitalCredential(const LocalFrame& frame, const DigitalCredentialRequestOptions& options, DigitalCredentialRequestCompletionHandler&& handler)
{
    RefPtr webFrame = WebFrame::fromCoreFrame(frame);
    RefPtr page = m_page.get();
    if (!webFrame || !page) {
        LOG_ERROR("Unable to get frame or page");
        handler(ExceptionData { ExceptionCode::InvalidStateError, "Unable to get frame or page"_s });
        return;
    }
    page->sendWithAsyncReply(Messages::DigitalCredentialsCoordinatorProxy::RequestDigitalCredential(webFrame->frameID(), webFrame->info(), options), WTFMove(handler));
}

void DigitalCredentialsCoordinator::cancel(CompletionHandler<void()>&& handler)
{
    RefPtr page = m_page.get();
    if (!page) {
        handler();
        return;
    }

    page->sendWithAsyncReply(Messages::DigitalCredentialsCoordinatorProxy::Cancel(), WTFMove(handler));
}

} // namespace WebKit

#endif // ENABLE(WEB_AUTHN)
