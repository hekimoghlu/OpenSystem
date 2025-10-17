/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 16, 2022.
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

#include <WebCore/AudioSession.h>
#include <WebCore/ProcessIdentifier.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakHashSet.h>
#include <wtf/WeakRef.h>

namespace WebKit {

class GPUProcess;
class RemoteAudioSessionProxy;

class RemoteAudioSessionProxyManager
    : public RefCounted<RemoteAudioSessionProxyManager>
    , public WebCore::AudioSessionInterruptionObserver
    , private WebCore::AudioSessionConfigurationChangeObserver {
    WTF_MAKE_TZONE_ALLOCATED(RemoteAudioSessionProxyManager);
public:
    static Ref<RemoteAudioSessionProxyManager> create(GPUProcess& gpuProcess)
    {
        return adoptRef(*new RemoteAudioSessionProxyManager(gpuProcess));
    }

    ~RemoteAudioSessionProxyManager();

    void addProxy(RemoteAudioSessionProxy&, std::optional<audit_token_t>);
    void removeProxy(RemoteAudioSessionProxy&);

    void updateCategory();
    void updatePreferredBufferSizeForProcess();
    void updateSpatialExperience();

    bool tryToSetActiveForProcess(RemoteAudioSessionProxy&, bool);

    void beginInterruptionRemote();
    void endInterruptionRemote(WebCore::AudioSession::MayResume);

    WebCore::AudioSession& session() { return WebCore::AudioSession::sharedSession(); }
    const WebCore::AudioSession& session() const { return WebCore::AudioSession::sharedSession(); }
    Ref<WebCore::AudioSession> protectedSession() { return WebCore::AudioSession::sharedSession(); }
    Ref<const WebCore::AudioSession> protectedSession() const { return WebCore::AudioSession::sharedSession(); }

    void updatePresentingProcesses();

    USING_CAN_MAKE_WEAKPTR(WebCore::AudioSessionInterruptionObserver);

private:
    RemoteAudioSessionProxyManager(GPUProcess&);

    void beginAudioSessionInterruption() final;
    void endAudioSessionInterruption(WebCore::AudioSession::MayResume) final;

    void hardwareMutedStateDidChange(const WebCore::AudioSession&) final;
    void bufferSizeDidChange(const WebCore::AudioSession&) final;
    void sampleRateDidChange(const WebCore::AudioSession&) final;
    void configurationDidChange(const WebCore::AudioSession&);

    bool hasOtherActiveProxyThan(RemoteAudioSessionProxy& proxyToExclude);
    bool hasActiveNotInterruptedProxy();

    WeakRef<GPUProcess> m_gpuProcess;
    WeakHashSet<RemoteAudioSessionProxy> m_proxies;
};

}

#endif
