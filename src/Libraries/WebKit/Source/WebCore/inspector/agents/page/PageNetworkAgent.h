/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 25, 2024.
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

#include "InspectorNetworkAgent.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class InspectorClient;
class Page;

class PageNetworkAgent final : public InspectorNetworkAgent {
    WTF_MAKE_NONCOPYABLE(PageNetworkAgent);
    WTF_MAKE_TZONE_ALLOCATED(PageNetworkAgent);
public:
    PageNetworkAgent(PageAgentContext&, InspectorClient*);
    ~PageNetworkAgent();

private:
    Inspector::Protocol::Network::LoaderId loaderIdentifier(DocumentLoader*);
    Inspector::Protocol::Network::FrameId frameIdentifier(DocumentLoader*);
    Vector<WebSocket*> activeWebSockets() WTF_REQUIRES_LOCK(WebSocket::allActiveWebSocketsLock());
    void setResourceCachingDisabledInternal(bool);
#if ENABLE(INSPECTOR_NETWORK_THROTTLING)
    bool setEmulatedConditionsInternal(std::optional<int>&& bytesPerSecondLimit);
#endif
    ScriptExecutionContext* scriptExecutionContext(Inspector::Protocol::ErrorString&, const Inspector::Protocol::Network::FrameId&);
    void addConsoleMessage(std::unique_ptr<Inspector::ConsoleMessage>&&);
    bool shouldForceBufferingNetworkResourceData() const { return false; }

    WeakRef<Page> m_inspectedPage;
#if ENABLE(INSPECTOR_NETWORK_THROTTLING)
    InspectorClient* m_client { nullptr };
#endif
};

} // namespace WebCore
