/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 13, 2025.
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

#if ENABLE(CONTENT_FILTERING)

#include <functional>
#include <wtf/RetainPtr.h>
#include <wtf/URL.h>
#include <wtf/text/WTFString.h>

OBJC_CLASS NSCoder;
OBJC_CLASS NSNumber;

#if PLATFORM(IOS_FAMILY)
OBJC_CLASS WebFilterEvaluator;
#endif

namespace WebCore {

class ResourceRequest;

class ContentFilterUnblockHandler {
public:
    using DecisionHandlerFunction = std::function<void(bool unblocked)>;
    using UnblockRequesterFunction = std::function<void(DecisionHandlerFunction)>;

    ContentFilterUnblockHandler() = default;
    WEBCORE_EXPORT ContentFilterUnblockHandler(String unblockURLHost, UnblockRequesterFunction);
#if HAVE(PARENTAL_CONTROLS_WITH_UNBLOCK_HANDLER)
    ContentFilterUnblockHandler(String unblockURLHost, RetainPtr<WebFilterEvaluator>);
#endif

    WEBCORE_EXPORT ContentFilterUnblockHandler(
        String&& unblockURLHost,
        URL&& unreachableURL,
#if HAVE(PARENTAL_CONTROLS_WITH_UNBLOCK_HANDLER)
        Vector<uint8_t>&& webFilterEvaluatorData,
#endif
        bool unblockedAfterRequest
    );

    WEBCORE_EXPORT bool needsUIProcess() const;
    WEBCORE_EXPORT bool canHandleRequest(const ResourceRequest&) const;
    WEBCORE_EXPORT void requestUnblockAsync(DecisionHandlerFunction) const;
    void wrapWithDecisionHandler(const DecisionHandlerFunction&);

    const String& unblockURLHost() const { return m_unblockURLHost; }
    const URL& unreachableURL() const { return m_unreachableURL; }
    void setUnreachableURL(const URL& url) { m_unreachableURL = url; }

#if HAVE(PARENTAL_CONTROLS_WITH_UNBLOCK_HANDLER)
    WEBCORE_EXPORT Vector<uint8_t> webFilterEvaluatorData() const;
#endif

    WEBCORE_EXPORT void setUnblockedAfterRequest(bool);
    bool unblockedAfterRequest() const { return m_unblockedAfterRequest; }

private:
#if HAVE(PARENTAL_CONTROLS_WITH_UNBLOCK_HANDLER)
    static RetainPtr<WebFilterEvaluator> unpackWebFilterEvaluatorData(Vector<uint8_t>&&);
#endif

    String m_unblockURLHost;
    URL m_unreachableURL;
    UnblockRequesterFunction m_unblockRequester;
#if HAVE(PARENTAL_CONTROLS_WITH_UNBLOCK_HANDLER)
    RetainPtr<WebFilterEvaluator> m_webFilterEvaluator;
#endif
    bool m_unblockedAfterRequest { false };
};

} // namespace WebCore

#endif // ENABLE(CONTENT_FILTERING)
