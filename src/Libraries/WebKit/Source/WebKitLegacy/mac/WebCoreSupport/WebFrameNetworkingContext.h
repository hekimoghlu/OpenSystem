/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 21, 2024.
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

#include <WebCore/FrameNetworkingContext.h>

class WebFrameNetworkingContext : public WebCore::FrameNetworkingContext {
public:
    static Ref<WebFrameNetworkingContext> create(WebCore::LocalFrame* frame)
    {
        return adoptRef(*new WebFrameNetworkingContext(frame));
    }

    static WebCore::NetworkStorageSession& ensurePrivateBrowsingSession();
    static void destroyPrivateBrowsingSession();

private:

    WebFrameNetworkingContext(WebCore::LocalFrame* frame)
        : WebCore::FrameNetworkingContext(frame)
    {
    }

    bool localFileContentSniffingEnabled() const override;
    SchedulePairHashSet* scheduledRunLoopPairs() const override;
    RetainPtr<CFDataRef> sourceApplicationAuditData() const override;
    String sourceApplicationIdentifier() const override;
    WebCore::ResourceError blockedError(const WebCore::ResourceRequest&) const override;
    WebCore::NetworkStorageSession* storageSession() const override;
};
