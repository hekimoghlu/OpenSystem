/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 21, 2025.
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

#if ENABLE(MEDIA_STREAM)

#include "MessageReceiver.h"
#include "MessageSender.h"
#include "SandboxExtension.h"
#include <WebCore/PageIdentifier.h>
#include <WebCore/RealtimeMediaSourceIdentifier.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {
class CaptureDevice;
}

namespace WebKit {

class WebProcess;

class SpeechRecognitionRealtimeMediaSourceManager final : public IPC::MessageReceiver, private IPC::MessageSender {
    WTF_MAKE_TZONE_ALLOCATED(SpeechRecognitionRealtimeMediaSourceManager);
public:
    explicit SpeechRecognitionRealtimeMediaSourceManager(WebProcess&);
    ~SpeechRecognitionRealtimeMediaSourceManager();

    void ref() const final;
    void deref() const final;

private:
    // Messages::SpeechRecognitionRealtimeMediaSourceManager
    void createSource(WebCore::RealtimeMediaSourceIdentifier, const WebCore::CaptureDevice&, WebCore::PageIdentifier);
    void deleteSource(WebCore::RealtimeMediaSourceIdentifier);
    void start(WebCore::RealtimeMediaSourceIdentifier);
    void stop(WebCore::RealtimeMediaSourceIdentifier);

    // IPC::MessageReceiver.
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) final;

    // IPC::MessageSender.
    IPC::Connection* messageSenderConnection() const final;
    uint64_t messageSenderDestinationID() const final;

    IPC::Connection& connection() const;
    Ref<IPC::Connection> protectedConnection() const;

    WeakRef<WebProcess> m_process;

    class Source;
    friend class Source;
    HashMap<WebCore::RealtimeMediaSourceIdentifier, std::unique_ptr<Source>> m_sources;

#if ENABLE(SANDBOX_EXTENSIONS)
    RefPtr<SandboxExtension> m_machBootstrapExtension;
    RefPtr<SandboxExtension> m_sandboxExtensionForTCCD;
    RefPtr<SandboxExtension> m_sandboxExtensionForMicrophone;
#endif
};

} // namespace WebKit

#endif
