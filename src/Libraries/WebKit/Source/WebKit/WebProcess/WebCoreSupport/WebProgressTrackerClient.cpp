/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 12, 2024.
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
#include "WebProgressTrackerClient.h"

#include "APIInjectedBundlePageLoaderClient.h"
#include "MessageSenderInlines.h"
#include "WebPage.h"
#include "WebPageProxyMessages.h"
#include <WebCore/LocalFrame.h>
#include <WebCore/Page.h>
#include <WebCore/ProgressTracker.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {
using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebProgressTrackerClient);

WebProgressTrackerClient::WebProgressTrackerClient(WebPage& webPage)
    : m_webPage(webPage)
{
}

void WebProgressTrackerClient::progressStarted(LocalFrame& originatingProgressFrame)
{
    if (!originatingProgressFrame.isMainFrame())
        return;

    Ref page = *m_webPage;
    page->setMainFrameProgressCompleted(false);
    page->send(Messages::WebPageProxy::DidStartProgress());
}

void WebProgressTrackerClient::progressEstimateChanged(LocalFrame& originatingProgressFrame)
{
    if (!originatingProgressFrame.isMainFrame())
        return;

    Ref page = *m_webPage;
    double progress = page->corePage()->progress().estimatedProgress();
    page->send(Messages::WebPageProxy::DidChangeProgress(progress));
}

void WebProgressTrackerClient::progressFinished(LocalFrame& originatingProgressFrame)
{
    if (!originatingProgressFrame.isMainFrame())
        return;

    Ref webPage = *m_webPage;
    webPage->setMainFrameProgressCompleted(true);

    // Notify the bundle client.
    webPage->injectedBundleLoaderClient().didFinishProgress(webPage);

    webPage->send(Messages::WebPageProxy::DidFinishProgress());
}

} // namespace WebKit
