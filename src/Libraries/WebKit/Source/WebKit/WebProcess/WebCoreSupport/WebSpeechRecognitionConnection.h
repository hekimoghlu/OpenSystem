/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 8, 2024.
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

#include "MessageReceiver.h"
#include "MessageSender.h"
#include <WebCore/PageIdentifier.h>
#include <WebCore/SpeechRecognitionConnection.h>

namespace WebCore {

class SpeechRecognitionConnectionClient;
class SpeechRecognitionUpdate;

}

namespace WebKit {

class WebPage;

using SpeechRecognitionConnectionIdentifier = WebCore::PageIdentifier;

class WebSpeechRecognitionConnection final : public WebCore::SpeechRecognitionConnection, private IPC::MessageReceiver, private IPC::MessageSender {
public:
    static Ref<WebSpeechRecognitionConnection> create(SpeechRecognitionConnectionIdentifier);

    void ref() const final { WebCore::SpeechRecognitionConnection::ref(); }
    void deref() const final { WebCore::SpeechRecognitionConnection::deref(); }

    void start(WebCore::SpeechRecognitionConnectionClientIdentifier, const String& lang, bool continuous, bool interimResults, uint64_t maxAlternatives, WebCore::ClientOrigin&&, WebCore::FrameIdentifier) final;
    void stop(WebCore::SpeechRecognitionConnectionClientIdentifier) final;
    void abort(WebCore::SpeechRecognitionConnectionClientIdentifier) final;

private:
    explicit WebSpeechRecognitionConnection(SpeechRecognitionConnectionIdentifier);
    ~WebSpeechRecognitionConnection();

    void registerClient(WebCore::SpeechRecognitionConnectionClient&) final;
    void unregisterClient(WebCore::SpeechRecognitionConnectionClient&) final;
    void didReceiveUpdate(WebCore::SpeechRecognitionUpdate&&) final;
    void invalidate(WebCore::SpeechRecognitionConnectionClientIdentifier);

    // IPC::MessageReceiver.
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) override;

    // IPC::MessageSender.
    IPC::Connection* messageSenderConnection() const final;
    uint64_t messageSenderDestinationID() const final;

    SpeechRecognitionConnectionIdentifier m_identifier;
    HashMap<WebCore::SpeechRecognitionConnectionClientIdentifier, WeakPtr<WebCore::SpeechRecognitionConnectionClient>> m_clientMap;
};

} // namespace WebKit
