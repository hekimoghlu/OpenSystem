/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 18, 2023.
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

#include "NetworkResourceLoader.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {
class LinkHeader;
}

namespace WebKit {

class EarlyHintsResourceLoader
    : public WebCore::ContentSecurityPolicyClient {
    WTF_MAKE_TZONE_ALLOCATED(EarlyHintsResourceLoader);
    WTF_MAKE_NONCOPYABLE(EarlyHintsResourceLoader);
public:
    explicit EarlyHintsResourceLoader(NetworkResourceLoader&);
    virtual ~EarlyHintsResourceLoader();

    void handleEarlyHintsResponse(WebCore::ResourceResponse&&);

private:
    // ContentSecurityPolicyClient
    void addConsoleMessage(MessageSource, MessageLevel, const String&, unsigned long requestIdentifier = 0) final;
    void enqueueSecurityPolicyViolationEvent(WebCore::SecurityPolicyViolationEventInit&&) final;

    WebCore::ResourceRequest constructPreconnectRequest(const WebCore::ResourceRequest&, const URL&);
    void startPreconnectTask(const URL& baseURL, const WebCore::LinkHeader&, const WebCore::ContentSecurityPolicy&);

    WeakPtr<NetworkResourceLoader> m_loader;
    bool m_hasReceivedEarlyHints { false };
};

} // namespace WebKit
