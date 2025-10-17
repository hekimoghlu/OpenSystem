/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 11, 2023.
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
#pragma once

#include <WebCore/FrameIdentifier.h>
#include <wtf/CompletionHandler.h>

namespace WebKit {

class WebFrameProxy;
class WebPageProxy;

enum class FindOptions : uint16_t;

class FindStringCallbackAggregator : public RefCounted<FindStringCallbackAggregator> {
public:
    static Ref<FindStringCallbackAggregator> create(WebPageProxy&, const String&, OptionSet<FindOptions>, unsigned maxMatchCount, CompletionHandler<void(bool)>&&);
    void foundString(std::optional<WebCore::FrameIdentifier>, uint32_t matchCount, bool didWrap);
    ~FindStringCallbackAggregator();

private:
    FindStringCallbackAggregator(WebPageProxy&, const String&, OptionSet<FindOptions>, unsigned maxMatchCount, CompletionHandler<void(bool)>&&);

    RefPtr<WebFrameProxy> incrementFrame(WebFrameProxy&);
    bool shouldTargetFrame(WebFrameProxy&, WebFrameProxy& focusedFrame, bool didWrap);

    WeakPtr<WebPageProxy> m_page;
    String m_string;
    OptionSet<FindOptions> m_options;
    unsigned m_maxMatchCount;
    uint32_t m_matchCount { 0 };
    CompletionHandler<void(bool)> m_completionHandler;
    HashMap<WebCore::FrameIdentifier, bool> m_matches;
};

} // namespace WebKit
