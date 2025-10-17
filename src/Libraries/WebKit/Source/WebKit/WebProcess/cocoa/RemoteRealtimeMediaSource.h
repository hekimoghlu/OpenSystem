/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 18, 2024.
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

#if PLATFORM(COCOA) && ENABLE(MEDIA_STREAM)

#include "GPUProcessConnection.h"
#include "RemoteRealtimeMediaSourceProxy.h"
#include <WebCore/RealtimeMediaSource.h>
#include <wtf/CheckedRef.h>

namespace WebKit {

class UserMediaCaptureManager;

class RemoteRealtimeMediaSource : public WebCore::RealtimeMediaSource
#if ENABLE(GPU_PROCESS)
    , public GPUProcessConnection::Client
#endif
    , public ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<RemoteRealtimeMediaSource, WTF::DestructionThread::MainRunLoop>
{
public:
    ~RemoteRealtimeMediaSource();

    WebCore::RealtimeMediaSourceIdentifier identifier() const { return m_proxy.identifier(); }
    IPC::Connection& connection() { return m_proxy.connection(); }

    void setSettings(WebCore::RealtimeMediaSourceSettings&&);

    void applyConstraintsSucceeded(WebCore::RealtimeMediaSourceSettings&&);
    void applyConstraintsFailed(WebCore::MediaConstraintType invalidConstraint, String&& errorMessage) { m_proxy.applyConstraintsFailed(invalidConstraint, WTFMove(errorMessage)); }

    void captureStopped(bool didFail);
    void sourceMutedChanged(bool value, bool interrupted);

    void configurationChanged(String&& persistentID, WebCore::RealtimeMediaSourceSettings&&, WebCore::RealtimeMediaSourceCapabilities&&);

#if ENABLE(GPU_PROCESS)
    WTF_ABSTRACT_THREAD_SAFE_REF_COUNTED_AND_CAN_MAKE_WEAK_PTR_IMPL;
#endif

protected:
    RemoteRealtimeMediaSource(WebCore::RealtimeMediaSourceIdentifier, const WebCore::CaptureDevice&, const WebCore::MediaConstraints*, WebCore::MediaDeviceHashSalts&&, UserMediaCaptureManager&, bool shouldCaptureInGPUProcess, std::optional<WebCore::PageIdentifier>);
    RemoteRealtimeMediaSource(RemoteRealtimeMediaSourceProxy&&, WebCore::MediaDeviceHashSalts&&, UserMediaCaptureManager&, std::optional<WebCore::PageIdentifier>);
    void createRemoteMediaSource();

    RemoteRealtimeMediaSourceProxy& proxy() { return m_proxy; }
    UserMediaCaptureManager& manager();

    void setCapabilities(WebCore::RealtimeMediaSourceCapabilities&&);

    const WebCore::RealtimeMediaSourceSettings& settings() final { return m_settings; }
    const WebCore::RealtimeMediaSourceCapabilities& capabilities() final { return m_capabilities; }

    Ref<TakePhotoNativePromise> takePhoto(WebCore::PhotoSettings&&) final;
    Ref<PhotoCapabilitiesNativePromise> getPhotoCapabilities() final;
    Ref<PhotoSettingsNativePromise> getPhotoSettings() final;

private:
    // RealtimeMediaSource
    void startProducingData() final { m_proxy.startProducingData(*pageIdentifier()); }
    void stopProducingData() final { m_proxy.stopProducingData(); }
    void endProducingData() final { m_proxy.endProducingData(); }
    bool isCaptureSource() const final { return true; }
    void applyConstraints(const WebCore::MediaConstraints&, ApplyConstraintsHandler&&) final;
    void didEnd() final;
    void whenReady(CompletionHandler<void(WebCore::CaptureSourceError&&)>&& callback) final { m_proxy.whenReady(WTFMove(callback)); }
    WebCore::CaptureDevice::DeviceType deviceType() const final { return m_proxy.deviceType(); }
    bool interrupted() const final { return m_proxy.interrupted(); }
    bool isPowerEfficient() const final { return m_proxy.isPowerEfficient(); }

#if ENABLE(GPU_PROCESS)
    // GPUProcessConnection::Client
    void gpuProcessConnectionDidClose(GPUProcessConnection&) final;
#endif

    RemoteRealtimeMediaSourceProxy m_proxy;
    CheckedRef<UserMediaCaptureManager> m_manager;
    std::optional<WebCore::MediaConstraints> m_constraints;
    WebCore::RealtimeMediaSourceCapabilities m_capabilities;
    std::optional<WebCore::PhotoCapabilities> m_photoCapabilities;
    WebCore::RealtimeMediaSourceSettings m_settings;
};

inline void RemoteRealtimeMediaSource::sourceMutedChanged(bool muted, bool interrupted)
{
    m_proxy.setInterrupted(interrupted);
    notifyMutedChange(muted);
}

} // namespace WebKit

#endif
