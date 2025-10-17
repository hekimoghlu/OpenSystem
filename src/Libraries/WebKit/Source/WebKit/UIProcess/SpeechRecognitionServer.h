/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 24, 2025.
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
#include "SpeechRecognitionPermissionRequest.h"
#include <WebCore/PageIdentifier.h>
#include <WebCore/SpeechRecognitionError.h>
#include <WebCore/SpeechRecognitionRequest.h>
#include <WebCore/SpeechRecognitionResultData.h>
#include <WebCore/SpeechRecognizer.h>
#include <wtf/Deque.h>
#include <wtf/RefCounted.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakRef.h>

namespace WebCore {
enum class SpeechRecognitionUpdateType : uint8_t;
struct CaptureSourceOrError;
struct ClientOrigin;
}

namespace WebKit {

class WebProcessProxy;
struct SharedPreferencesForWebProcess;

using SpeechRecognitionServerIdentifier = WebCore::PageIdentifier;
using SpeechRecognitionPermissionChecker = Function<void(WebCore::SpeechRecognitionRequest&, SpeechRecognitionPermissionRequestCallback&&)>;
using SpeechRecognitionCheckIfMockSpeechRecognitionEnabled = Function<bool()>;

class SpeechRecognitionServer : public IPC::MessageReceiver, private IPC::MessageSender, public RefCounted<SpeechRecognitionServer> {
    WTF_MAKE_TZONE_ALLOCATED(SpeechRecognitionServer);
public:
    using RealtimeMediaSourceCreateFunction = Function<WebCore::CaptureSourceOrError()>;
    static Ref<SpeechRecognitionServer> create(WebProcessProxy&, SpeechRecognitionServerIdentifier, SpeechRecognitionPermissionChecker&&, SpeechRecognitionCheckIfMockSpeechRecognitionEnabled&&
#if ENABLE(MEDIA_STREAM)
        , RealtimeMediaSourceCreateFunction&&
#endif
        );

    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    std::optional<SharedPreferencesForWebProcess> sharedPreferencesForWebProcess() const;

    void start(WebCore::SpeechRecognitionConnectionClientIdentifier, String&& lang, bool continuous, bool interimResults, uint64_t maxAlternatives, WebCore::ClientOrigin&&, WebCore::FrameIdentifier);
    void stop(WebCore::SpeechRecognitionConnectionClientIdentifier);
    void abort(WebCore::SpeechRecognitionConnectionClientIdentifier);
    void invalidate(WebCore::SpeechRecognitionConnectionClientIdentifier);
    void mute();

private:
    SpeechRecognitionServer(WebProcessProxy&, SpeechRecognitionServerIdentifier, SpeechRecognitionPermissionChecker&&, SpeechRecognitionCheckIfMockSpeechRecognitionEnabled&&
#if ENABLE(MEDIA_STREAM)
        , RealtimeMediaSourceCreateFunction&&
#endif
        );

    void requestPermissionForRequest(WebCore::SpeechRecognitionRequest&);
    void handleRequest(UniqueRef<WebCore::SpeechRecognitionRequest>&&);
    void sendUpdate(WebCore::SpeechRecognitionConnectionClientIdentifier, WebCore::SpeechRecognitionUpdateType, std::optional<WebCore::SpeechRecognitionError> = std::nullopt, std::optional<Vector<WebCore::SpeechRecognitionResultData>> = std::nullopt);
    void sendUpdate(const WebCore::SpeechRecognitionUpdate&);

    // IPC::MessageReceiver.
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) override;

    // IPC::MessageSender.
    IPC::Connection* messageSenderConnection() const final;
    uint64_t messageSenderDestinationID() const final;

    WeakPtr<WebProcessProxy> m_process;
    SpeechRecognitionServerIdentifier m_identifier;
    HashMap<WebCore::SpeechRecognitionConnectionClientIdentifier, std::unique_ptr<WebCore::SpeechRecognitionRequest>> m_requests;
    SpeechRecognitionPermissionChecker m_permissionChecker;
    std::unique_ptr<WebCore::SpeechRecognizer> m_recognizer;
    SpeechRecognitionCheckIfMockSpeechRecognitionEnabled m_checkIfMockSpeechRecognitionEnabled;
    bool m_isResetting { false };

#if ENABLE(MEDIA_STREAM)
    RealtimeMediaSourceCreateFunction m_realtimeMediaSourceCreateFunction;
#endif
};

} // namespace WebKit
