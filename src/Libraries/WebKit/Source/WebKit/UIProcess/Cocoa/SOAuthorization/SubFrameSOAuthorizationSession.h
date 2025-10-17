/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 15, 2023.
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

#if HAVE(APP_SSO)

#include <variant>
#include "FrameLoadState.h"
#include "NavigationSOAuthorizationSession.h"
#include <WebCore/FrameIdentifier.h>
#include <wtf/Deque.h>

namespace WebKit {

class SubFrameSOAuthorizationSession final : public NavigationSOAuthorizationSession, public FrameLoadStateObserver {
public:
    using Callback = CompletionHandler<void(bool)>;

    static Ref<SOAuthorizationSession> create(RetainPtr<WKSOAuthorizationDelegate>, Ref<API::NavigationAction>&&, WebPageProxy&, Callback&&, std::optional<WebCore::FrameIdentifier>);

    ~SubFrameSOAuthorizationSession();

    void ref() const final { ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr::ref(); }
    void deref() const final { ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr::deref(); }

private:
    using Supplement = std::variant<Vector<uint8_t>, String>;

    SubFrameSOAuthorizationSession(RetainPtr<WKSOAuthorizationDelegate>, Ref<API::NavigationAction>&&, WebPageProxy&, Callback&&, std::optional<WebCore::FrameIdentifier>);

    // SOAuthorizationSession
    void fallBackToWebPathInternal() final;
    void abortInternal() final;
    void completeInternal(const WebCore::ResourceResponse&, NSData *) final;

    // NavigationSOAuthorizationSession
    void beforeStart() final;

    // FrameLoadStateObserver
    void didFinishLoad(IsMainFrame, const URL&) final;

    void appendRequestToLoad(URL&&, Supplement&&);
    void loadRequestToFrame();

    bool shouldInterruptLoadForXFrameOptions(Vector<Ref<WebCore::SecurityOrigin>>&& frameAncestorOrigins, const String& xFrameOptions, const URL&);
    bool shouldInterruptLoadForCSPFrameAncestorsOrXFrameOptions(const WebCore::ResourceResponse&) final;

    Markable<WebCore::FrameIdentifier> m_frameID;
    Deque<std::pair<URL, Supplement>> m_requestsToLoad;
};

} // namespace WebKit

#endif
