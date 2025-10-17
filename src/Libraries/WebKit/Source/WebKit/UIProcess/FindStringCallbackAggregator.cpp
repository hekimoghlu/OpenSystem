/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 15, 2024.
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
#include "FindStringCallbackAggregator.h"

#include "APIFindClient.h"
#include "WebFrameProxy.h"
#include "WebPageMessages.h"
#include "WebPageProxy.h"

namespace WebKit {

using namespace WebCore;

Ref<FindStringCallbackAggregator> FindStringCallbackAggregator::create(WebPageProxy& page, const String& string, OptionSet<FindOptions> options, unsigned maxMatchCount, CompletionHandler<void(bool)>&& completionHandler)
{
    return adoptRef(*new FindStringCallbackAggregator(page, string, options, maxMatchCount, WTFMove(completionHandler)));
}

void FindStringCallbackAggregator::foundString(std::optional<FrameIdentifier> frameID, uint32_t matchCount, bool didWrap)
{
    if (!frameID)
        return;

    m_matchCount += matchCount;
    m_matches.set(*frameID, didWrap);
}

RefPtr<WebFrameProxy> FindStringCallbackAggregator::incrementFrame(WebFrameProxy& frame)
{
    auto canWrap = m_options.contains(FindOptions::WrapAround) ? CanWrap::Yes : CanWrap::No;
    return m_options.contains(FindOptions::Backwards)
        ? frame.traversePrevious(canWrap).frame
        : frame.traverseNext(canWrap).frame;
}

bool FindStringCallbackAggregator::shouldTargetFrame(WebFrameProxy& frame, WebFrameProxy& focusedFrame, bool didWrap)
{
    if (!didWrap)
        return true;

    if (frame.process() != focusedFrame.process())
        return true;

    RefPtr nextFrameInProcess = incrementFrame(focusedFrame);
    while (nextFrameInProcess && nextFrameInProcess != &focusedFrame && nextFrameInProcess->process() == focusedFrame.process()) {
        if (nextFrameInProcess == &frame)
            return true;
        nextFrameInProcess = incrementFrame(*nextFrameInProcess);
    }
    return false;
}

FindStringCallbackAggregator::~FindStringCallbackAggregator()
{
    RefPtr protectedPage = m_page.get();
    if (!protectedPage) {
        m_completionHandler(false);
        return;
    }

    RefPtr focusedFrame = protectedPage->focusedOrMainFrame();
    if (!focusedFrame) {
        m_completionHandler(false);
        return;
    }

    RefPtr frameContainingMatch = focusedFrame.get();
    do {
        auto it = m_matches.find(frameContainingMatch->frameID());
        if (it != m_matches.end()) {
            if (shouldTargetFrame(*frameContainingMatch, *focusedFrame, it->value))
                break;
        }
        frameContainingMatch = incrementFrame(*frameContainingMatch);
    } while (frameContainingMatch && frameContainingMatch != focusedFrame);

    auto message = Messages::WebPage::FindString(m_string, m_options, m_maxMatchCount);
    auto completionHandler = [protectedPage = Ref { *protectedPage }, string = m_string, matchCount = m_matchCount, completionHandler = WTFMove(m_completionHandler)](std::optional<FrameIdentifier> frameID, Vector<IntRect>&& matchRects, uint32_t, int32_t matchIndex, bool didWrap) mutable {
        if (!frameID)
            protectedPage->findClient().didFailToFindString(protectedPage.ptr(), string);
        else
            protectedPage->findClient().didFindString(protectedPage.ptr(), string, matchRects, matchCount, matchIndex, didWrap);
        completionHandler(frameID.has_value());
    };

    Ref targetFrame = frameContainingMatch ? *frameContainingMatch : *focusedFrame;
    targetFrame->protectedProcess()->sendWithAsyncReply(WTFMove(message), WTFMove(completionHandler), protectedPage->webPageIDInProcess(targetFrame->protectedProcess()));
    if (frameContainingMatch && focusedFrame && focusedFrame->process() != frameContainingMatch->process())
        protectedPage->clearSelection(focusedFrame->frameID());
}

FindStringCallbackAggregator::FindStringCallbackAggregator(WebPageProxy& page, const String& string, OptionSet<FindOptions> options, unsigned maxMatchCount, CompletionHandler<void(bool)>&& completionHandler)
    : m_page(page)
    , m_string(string)
    , m_options(options)
    , m_maxMatchCount(maxMatchCount)
    , m_completionHandler(WTFMove(completionHandler))
{
}

} // namespace WebKit
