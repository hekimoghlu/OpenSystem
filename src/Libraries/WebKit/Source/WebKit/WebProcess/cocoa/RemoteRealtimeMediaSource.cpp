/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 17, 2024.
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
#include "config.h"
#include "RemoteRealtimeMediaSource.h"

#if PLATFORM(COCOA) && ENABLE(MEDIA_STREAM)

#include "GPUProcessConnection.h"
#include "UserMediaCaptureManager.h"
#include "WebProcess.h"
#include <WebCore/MediaConstraints.h>

namespace WebKit {
using namespace WebCore;

RemoteRealtimeMediaSource::RemoteRealtimeMediaSource(RealtimeMediaSourceIdentifier identifier, const CaptureDevice& device, const MediaConstraints* constraints, MediaDeviceHashSalts&& hashSalts, UserMediaCaptureManager& manager, bool shouldCaptureInGPUProcess, std::optional<PageIdentifier> pageIdentifier)
    : RealtimeMediaSource(device, WTFMove(hashSalts), pageIdentifier)
    , m_proxy(identifier, device, shouldCaptureInGPUProcess, constraints)
    , m_manager(manager)
{
}

RemoteRealtimeMediaSource::RemoteRealtimeMediaSource(RemoteRealtimeMediaSourceProxy&& proxy, MediaDeviceHashSalts&& hashSalts, UserMediaCaptureManager& manager, std::optional<PageIdentifier> pageIdentifier)
    : RealtimeMediaSource(proxy.device(), WTFMove(hashSalts), pageIdentifier)
    , m_proxy(WTFMove(proxy))
    , m_manager(manager)
{
}

RemoteRealtimeMediaSource::~RemoteRealtimeMediaSource() = default;

UserMediaCaptureManager& RemoteRealtimeMediaSource::manager()
{
    return m_manager.get();
}

void RemoteRealtimeMediaSource::createRemoteMediaSource()
{
    m_proxy.createRemoteMediaSource(deviceIDHashSalts(), *pageIdentifier(), [this, protectedThis = Ref { *this }](WebCore::CaptureSourceError&& error, WebCore::RealtimeMediaSourceSettings&& settings, WebCore::RealtimeMediaSourceCapabilities&& capabilities) {
        if (error) {
            m_proxy.didFail(WTFMove(error));
            return;
        }

        setSettings(WTFMove(settings));
        setCapabilities(WTFMove(capabilities));
        setName(m_settings.label());

        m_proxy.setAsReady();
        if (m_proxy.shouldCaptureInGPUProcess())
            WebProcess::singleton().ensureGPUProcessConnection().addClient(*this);
    }, m_proxy.shouldCaptureInGPUProcess() && m_manager->shouldUseGPUProcessRemoteFrames());
}

void RemoteRealtimeMediaSource::setCapabilities(RealtimeMediaSourceCapabilities&& capabilities)
{
    m_capabilities = WTFMove(capabilities);
}

void RemoteRealtimeMediaSource::setSettings(RealtimeMediaSourceSettings&& settings)
{
    auto changed = m_settings.difference(settings);
    m_settings = WTFMove(settings);
    notifySettingsDidChangeObservers(changed);
}

Ref<RealtimeMediaSource::TakePhotoNativePromise> RemoteRealtimeMediaSource::takePhoto(PhotoSettings&& settings)
{
    return m_proxy.takePhoto(WTFMove(settings));
}

Ref<RealtimeMediaSource::PhotoCapabilitiesNativePromise> RemoteRealtimeMediaSource::getPhotoCapabilities()
{
    return m_proxy.getPhotoCapabilities();
}

Ref<RealtimeMediaSource::PhotoSettingsNativePromise> RemoteRealtimeMediaSource::getPhotoSettings()
{
    return m_proxy.getPhotoSettings();
}

void RemoteRealtimeMediaSource::configurationChanged(String&& persistentID, WebCore::RealtimeMediaSourceSettings&& settings, WebCore::RealtimeMediaSourceCapabilities&& capabilities)
{
    setPersistentId(WTFMove(persistentID));
    setSettings(WTFMove(settings));
    setCapabilities(WTFMove(capabilities));
    setName(m_settings.label());
    
    forEachObserver([](auto& observer) {
        observer.sourceConfigurationChanged();
    });
}

void RemoteRealtimeMediaSource::applyConstraintsSucceeded(WebCore::RealtimeMediaSourceSettings&& settings)
{
    setSettings(WTFMove(settings));
    m_proxy.applyConstraintsSucceeded();
}

void RemoteRealtimeMediaSource::didEnd()
{
    if (m_proxy.isEnded())
        return;

    m_proxy.end();
    m_manager->removeSource(identifier());
    m_manager->remoteCaptureSampleManager().removeSource(identifier());
}

void RemoteRealtimeMediaSource::captureStopped(bool didFail)
{
    if (didFail) {
        captureFailed();
        return;
    }
    end();
}

void RemoteRealtimeMediaSource::applyConstraints(const MediaConstraints& constraints, ApplyConstraintsHandler&& callback)
{
    m_constraints = constraints;
    m_proxy.applyConstraints(constraints, WTFMove(callback));
}

#if ENABLE(GPU_PROCESS)
void RemoteRealtimeMediaSource::gpuProcessConnectionDidClose(GPUProcessConnection&)
{
    ASSERT(m_proxy.shouldCaptureInGPUProcess());
    if (isEnded())
        return;

    m_proxy.updateConnection();
    m_manager->remoteCaptureSampleManager().didUpdateSourceConnection(Ref { m_proxy.connection() });
    m_proxy.resetReady();
    createRemoteMediaSource();

    m_proxy.failApplyConstraintCallbacks("GPU Process terminated"_s);
    if (m_constraints)
        m_proxy.applyConstraints(*m_constraints, [](auto) { });

    if (isProducingData())
        startProducingData();

}
#endif

}

#endif
