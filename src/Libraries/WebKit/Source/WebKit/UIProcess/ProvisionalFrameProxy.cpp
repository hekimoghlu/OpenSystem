/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 9, 2022.
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
#include "ProvisionalFrameProxy.h"

#include "FrameProcess.h"
#include "ProvisionalFrameCreationParameters.h"
#include "VisitedLinkStore.h"
#include "WebFrameProxy.h"
#include "WebPageMessages.h"
#include "WebPageProxy.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ProvisionalFrameProxy);

ProvisionalFrameProxy::ProvisionalFrameProxy(WebFrameProxy& frame, Ref<FrameProcess>&& frameProcess)
    : m_frame(frame)
    , m_frameProcess(WTFMove(frameProcess))
    , m_visitedLinkStore(frame.page()->visitedLinkStore())
{
    process().markProcessAsRecentlyUsed();
    process().send(Messages::WebPage::CreateProvisionalFrame(ProvisionalFrameCreationParameters {
        frame.layerHostingContextIdentifier(),
        frame.effectiveSandboxFlags(),
        frame.scrollingMode()
    }, frame.frameID()), frame.page()->webPageIDInProcess(process()));
}

ProvisionalFrameProxy::~ProvisionalFrameProxy()
{
    if (m_frameProcess && m_frame->page())
        process().send(Messages::WebPage::DestroyProvisionalFrame(m_frame->frameID()), m_frame->page()->webPageIDInProcess(process()));
}

RefPtr<FrameProcess> ProvisionalFrameProxy::takeFrameProcess()
{
    ASSERT(m_frameProcess);
    return std::exchange(m_frameProcess, nullptr).releaseNonNull();
}

WebProcessProxy& ProvisionalFrameProxy::process() const
{
    ASSERT(m_frameProcess);
    return m_frameProcess->process();
}

Ref<WebProcessProxy> ProvisionalFrameProxy::protectedProcess() const
{
    return process();
}

} // namespace WebKit
