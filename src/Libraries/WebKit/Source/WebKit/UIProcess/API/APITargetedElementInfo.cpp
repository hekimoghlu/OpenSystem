/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 14, 2025.
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
#include "APITargetedElementInfo.h"

#include "APIFrameTreeNode.h"
#include "FrameTreeNodeData.h"
#include "PageClient.h"
#include "WebFrameProxy.h"
#include "WebPageProxy.h"
#include <WebCore/ShareableBitmap.h>
#include <wtf/Box.h>
#include <wtf/CallbackAggregator.h>

namespace API {
using namespace WebKit;

TargetedElementInfo::TargetedElementInfo(WebPageProxy& page, WebCore::TargetedElementInfo&& info)
    : m_info(WTFMove(info))
    , m_page(page)
{
}

bool TargetedElementInfo::isSameElement(const TargetedElementInfo& other) const
{
    return m_info.elementIdentifier == other.m_info.elementIdentifier
        && m_info.documentIdentifier == other.m_info.documentIdentifier
        && m_page == other.m_page;
}

WebCore::FloatRect TargetedElementInfo::boundsInWebView() const
{
    RefPtr page = m_page.get();
    if (!page)
        return { };
    RefPtr pageClient = page->pageClient();
    if (!pageClient)
        return { };
    return pageClient->rootViewToWebView(boundsInRootView());
}

void TargetedElementInfo::childFrames(CompletionHandler<void(Vector<Ref<FrameTreeNode>>&&)>&& completion) const
{
    RefPtr page = m_page.get();
    if (!page)
        return completion({ });

    auto aggregateData = Box<Vector<FrameTreeNodeData>>::create();
    auto aggregator = CallbackAggregator::create([page, aggregateData, completion = WTFMove(completion)]() mutable {
        completion(WTF::map(WTFMove(*aggregateData), [&](auto&& data) {
            return FrameTreeNode::create(WTFMove(data), *page);
        }));
    });

    for (auto identifier : m_info.childFrameIdentifiers) {
        RefPtr frame = WebFrameProxy::webFrame(identifier);
        if (!frame)
            continue;

        if (frame->page() != page)
            continue;

        frame->getFrameInfo([aggregator, aggregateData](auto&& data) {
            aggregateData->append(WTFMove(data));
        });
    }
}

void TargetedElementInfo::takeSnapshot(CompletionHandler<void(std::optional<WebCore::ShareableBitmapHandle>&&)>&& completion)
{
    RefPtr page = m_page.get();
    if (!page)
        return completion({ });

    page->takeSnapshotForTargetedElement(*this, WTFMove(completion));
}

} // namespace API
