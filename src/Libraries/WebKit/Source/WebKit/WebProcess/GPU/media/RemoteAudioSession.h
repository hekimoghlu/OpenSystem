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

#if ENABLE(GPU_PROCESS) && USE(AUDIO_SESSION)

#include "GPUProcessConnection.h"
#include "MessageReceiver.h"
#include "RemoteAudioSessionConfiguration.h"
#include <WebCore/AudioSession.h>
#include <wtf/TZoneMalloc.h>

namespace IPC {
class Connection;
}

namespace WebKit {

class GPUProcessConnection;
class WebProcess;

class RemoteAudioSession final
    : public WebCore::AudioSession
    , public WebCore::AudioSessionInterruptionObserver
    , public GPUProcessConnection::Client
    , IPC::MessageReceiver {
    WTF_MAKE_TZONE_ALLOCATED(RemoteAudioSession);
public:
    static Ref<RemoteAudioSession> create();
    ~RemoteAudioSession();

    WTF_ABSTRACT_THREAD_SAFE_REF_COUNTED_AND_CAN_MAKE_WEAK_PTR_IMPL;

private:
    RemoteAudioSession();
    IPC::Connection& ensureConnection();

    // IPC::MessageReceiver
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) final;

    // Messages
    void configurationChanged(RemoteAudioSessionConfiguration&&);

    // GPUProcessConnection::Client
    void gpuProcessConnectionDidClose(GPUProcessConnection&) final;

    // AudioSession
    void setCategory(CategoryType, Mode, WebCore::RouteSharingPolicy) final;
    CategoryType category() const final;
    Mode mode() const final;

    WebCore::RouteSharingPolicy routeSharingPolicy() const final { return m_routeSharingPolicy; }
    String routingContextUID() const final { return configuration().routingContextUID; }

    float sampleRate() const final { return configuration().sampleRate; }
    size_t bufferSize() const final { return configuration().bufferSize; }
    size_t numberOfOutputChannels() const final { return configuration().numberOfOutputChannels; }
    size_t maximumNumberOfOutputChannels() const final { return configuration().maximumNumberOfOutputChannels; }
    size_t outputLatency() const final { return configuration().outputLatency; }

    bool tryToSetActiveInternal(bool) final;

    size_t preferredBufferSize() const final { return configuration().preferredBufferSize; }
    void setPreferredBufferSize(size_t) final;
        
    void addConfigurationChangeObserver(WebCore::AudioSessionConfigurationChangeObserver&) final;
    void removeConfigurationChangeObserver(WebCore::AudioSessionConfigurationChangeObserver&) final;

    void setIsPlayingToBluetoothOverride(std::optional<bool>) final;

    bool isMuted() const final { return configuration().isMuted; }

    bool isActive() const final { return configuration().isActive; }

    void beginInterruptionForTesting() final;
    void endInterruptionForTesting() final;
    void clearInterruptionFlagForTesting() final { m_isInterruptedForTesting = false; }

    void setSceneIdentifier(const String&) final;
    const String& sceneIdentifier() const final { return configuration().sceneIdentifier; }

    void setSoundStageSize(SoundStageSize) final;
    SoundStageSize soundStageSize() const final { return configuration().soundStageSize; }

    const RemoteAudioSessionConfiguration& configuration() const;
    RemoteAudioSessionConfiguration& configuration();
    void initializeConfigurationIfNecessary();

    void beginInterruptionRemote();
    void endInterruptionRemote(MayResume);

    // InterruptionObserver
    void beginAudioSessionInterruption() final;
    void endAudioSessionInterruption(MayResume) final;

    WeakHashSet<WebCore::AudioSessionConfigurationChangeObserver> m_configurationChangeObservers;
    CategoryType m_category { CategoryType::None };
    Mode m_mode { Mode::Default };
    WebCore::RouteSharingPolicy m_routeSharingPolicy { WebCore::RouteSharingPolicy::Default };
    bool m_isPlayingToBluetoothOverrideChanged { false };
    std::optional<RemoteAudioSessionConfiguration> m_configuration;
    ThreadSafeWeakPtr<GPUProcessConnection> m_gpuProcessConnection;
    bool m_isInterruptedForTesting { false };
};

}

#endif
