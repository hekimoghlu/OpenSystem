/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 9, 2022.
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

#include <WebCore/MessagePortChannelProvider.h>
#include <WebCore/MessagePortIdentifier.h>
#include <WebCore/MessageWithMessagePorts.h>
#include <wtf/TZoneMalloc.h>

namespace WebKit {

class WebMessagePortChannelProvider final : public WebCore::MessagePortChannelProvider {
    WTF_MAKE_TZONE_ALLOCATED(WebMessagePortChannelProvider);
public:
    static WebMessagePortChannelProvider& singleton();

    void messagePortSentToRemote(const WebCore::MessagePortIdentifier&);

private:
    WebMessagePortChannelProvider();
    ~WebMessagePortChannelProvider() final;

    void createNewMessagePortChannel(const WebCore::MessagePortIdentifier& local, const WebCore::MessagePortIdentifier& remote) final;
    void entangleLocalPortInThisProcessToRemote(const WebCore::MessagePortIdentifier& local, const WebCore::MessagePortIdentifier& remote) final;
    void messagePortDisentangled(const WebCore::MessagePortIdentifier& local) final;
    void messagePortClosed(const WebCore::MessagePortIdentifier& local) final;
    void takeAllMessagesForPort(const WebCore::MessagePortIdentifier&, CompletionHandler<void(Vector<WebCore::MessageWithMessagePorts>&&, CompletionHandler<void()>&&)>&&) final;
    void postMessageToRemote(WebCore::MessageWithMessagePorts&&, const WebCore::MessagePortIdentifier& remoteTarget) final;

    HashMap<WebCore::MessagePortIdentifier, Vector<WebCore::MessageWithMessagePorts>> m_inProcessPortMessages;
};

} // namespace WebKit
