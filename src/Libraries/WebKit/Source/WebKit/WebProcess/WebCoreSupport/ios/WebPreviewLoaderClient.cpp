/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 27, 2022.
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
#include "WebPreviewLoaderClient.h"

#if USE(QUICK_LOOK)

#include "Logging.h"
#include "WebPage.h"
#include "WebPageProxyMessages.h"
#include "WebProcess.h"
#include <wtf/Function.h>
#include <wtf/HashMap.h>
#include <wtf/NeverDestroyed.h>

namespace WebKit {

using namespace WebCore;

WebPreviewLoaderClient::WebPreviewLoaderClient(const String& fileName, const String& uti, PageIdentifier pageID)
    : m_fileName { fileName }
    , m_uti { uti }
    , m_pageID { pageID }
{
}

WebPreviewLoaderClient::~WebPreviewLoaderClient() = default;

void WebPreviewLoaderClient::didReceiveData(const SharedBuffer& buffer)
{
    auto webPage = WebProcess::singleton().webPage(m_pageID);
    if (!webPage)
        return;

    if (m_buffer.isEmpty())
        webPage->didStartLoadForQuickLookDocumentInMainFrame(m_fileName, m_uti);

    m_buffer.append(buffer);
}

void WebPreviewLoaderClient::didFinishLoading()
{
    auto webPage = WebProcess::singleton().webPage(m_pageID);
    if (!webPage)
        return;

    webPage->didFinishLoadForQuickLookDocumentInMainFrame(m_buffer.take().get());
}

void WebPreviewLoaderClient::didFail()
{
    m_buffer.reset();
}

void WebPreviewLoaderClient::didRequestPassword(Function<void(const String&)>&& completionHandler)
{
    auto webPage = WebProcess::singleton().webPage(m_pageID);
    if (!webPage) {
        completionHandler({ });
        return;
    }

    webPage->requestPasswordForQuickLookDocumentInMainFrame(m_fileName, WTFMove(completionHandler));
}

} // namespace WebKit

#endif // USE(QUICK_LOOK)
