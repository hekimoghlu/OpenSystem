/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 3, 2023.
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
#include "UserMediaCaptureManager.h"

#if PLATFORM(COCOA) && ENABLE(MEDIA_STREAM)

#include "AudioMediaStreamTrackRendererInternalUnitManager.h"
#include "GPUProcessConnection.h"
#include "RemoteRealtimeAudioSource.h"
#include "RemoteRealtimeVideoSource.h"
#include "RemoteVideoFrameObjectHeapProxy.h"
#include "UserMediaCaptureManagerMessages.h"
#include "WebProcess.h"
#include <WebCore/AudioMediaStreamTrackRendererUnit.h>
#include <WebCore/DeprecatedGlobalSettings.h>
#include <WebCore/MockRealtimeMediaSourceCenter.h>
#include <WebCore/RealtimeMediaSourceCenter.h>
#include <wtf/Assertions.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {
using namespace PAL;
using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(UserMediaCaptureManager);

UserMediaCaptureManager::UserMediaCaptureManager(WebProcess& process)
    : m_process(process)
    , m_audioFactory(*this)
    , m_videoFactory(*this)
    , m_displayFactory(*this)
    , m_remoteCaptureSampleManager(*this)
{
    process.addMessageReceiver(Messages::UserMediaCaptureManager::messageReceiverName(), *this);
}

UserMediaCaptureManager::~UserMediaCaptureManager()
{
    RealtimeMediaSourceCenter::singleton().unsetAudioCaptureFactory(m_audioFactory);
    RealtimeMediaSourceCenter::singleton().unsetDisplayCaptureFactory(m_displayFactory);
    RealtimeMediaSourceCenter::singleton().unsetVideoCaptureFactory(m_videoFactory);
    m_process->removeMessageReceiver(Messages::UserMediaCaptureManager::messageReceiverName());
    m_remoteCaptureSampleManager.stopListeningForIPC();
}

void UserMediaCaptureManager::ref() const
{
    m_process->ref();
}

void UserMediaCaptureManager::deref() const
{
    m_process->deref();
}

ASCIILiteral UserMediaCaptureManager::supplementName()
{
    return "UserMediaCaptureManager"_s;
}

void UserMediaCaptureManager::setupCaptureProcesses(bool shouldCaptureAudioInUIProcess, bool shouldCaptureAudioInGPUProcess, bool shouldCaptureVideoInUIProcess, bool shouldCaptureVideoInGPUProcess, bool shouldCaptureDisplayInUIProcess, bool shouldCaptureDisplayInGPUProcess, bool shouldUseGPUProcessRemoteFrames)
{
    m_shouldUseGPUProcessRemoteFrames = shouldUseGPUProcessRemoteFrames;
    // FIXME(rdar://84278146): Adopt AVCaptureSession attribution API for camera access in the web process if shouldCaptureVideoInGPUProcess is false.
    MockRealtimeMediaSourceCenter::singleton().setMockAudioCaptureEnabled(!shouldCaptureAudioInUIProcess && !shouldCaptureAudioInGPUProcess);
    MockRealtimeMediaSourceCenter::singleton().setMockVideoCaptureEnabled(!shouldCaptureVideoInUIProcess && !shouldCaptureVideoInGPUProcess);
    MockRealtimeMediaSourceCenter::singleton().setMockDisplayCaptureEnabled(!shouldCaptureDisplayInUIProcess && !shouldCaptureDisplayInGPUProcess);

    m_audioFactory.setShouldCaptureInGPUProcess(shouldCaptureAudioInGPUProcess);
    m_videoFactory.setShouldCaptureInGPUProcess(shouldCaptureVideoInGPUProcess);
    m_displayFactory.setShouldCaptureInGPUProcess(shouldCaptureDisplayInGPUProcess);

    if (shouldCaptureAudioInUIProcess || shouldCaptureAudioInGPUProcess)
        WebCore::AudioMediaStreamTrackRendererInternalUnit::setCreateFunction(createRemoteAudioMediaStreamTrackRendererInternalUnitProxy);

    if (shouldCaptureAudioInUIProcess || shouldCaptureAudioInGPUProcess)
        RealtimeMediaSourceCenter::singleton().setAudioCaptureFactory(m_audioFactory);
    if (shouldCaptureVideoInUIProcess || shouldCaptureVideoInGPUProcess)
        RealtimeMediaSourceCenter::singleton().setVideoCaptureFactory(m_videoFactory);
    if (shouldCaptureDisplayInUIProcess || shouldCaptureDisplayInGPUProcess)
        RealtimeMediaSourceCenter::singleton().setDisplayCaptureFactory(m_displayFactory);
}

void UserMediaCaptureManager::addSource(Ref<RemoteRealtimeAudioSource>&& source)
{
    auto identifier = source->identifier();
    ASSERT(!m_sources.contains(identifier));
    m_sources.add(identifier, Source(WTFMove(source)));
}

void UserMediaCaptureManager::addSource(Ref<RemoteRealtimeVideoSource>&& source)
{
    auto identifier = source->identifier();
    ASSERT(!m_sources.contains(identifier));
    m_sources.add(identifier, Source(WTFMove(source)));
}

void UserMediaCaptureManager::removeSource(RealtimeMediaSourceIdentifier identifier)
{
    ASSERT(m_sources.contains(identifier));
    m_sources.remove(identifier);
}

void UserMediaCaptureManager::sourceStopped(RealtimeMediaSourceIdentifier identifier, bool didFail)
{
    auto iterator = m_sources.find(identifier);
    if (iterator == m_sources.end())
        return;

    switchOn(iterator->value, [didFail](Ref<RemoteRealtimeAudioSource>& source) {
        source->captureStopped(didFail);
    }, [didFail](Ref<RemoteRealtimeVideoSource>& source) {
        source->captureStopped(didFail);
    }, [](std::nullptr_t) { });
}

void UserMediaCaptureManager::sourceMutedChanged(RealtimeMediaSourceIdentifier identifier, bool muted, bool interrupted)
{
    auto iterator = m_sources.find(identifier);
    if (iterator == m_sources.end())
        return;

    switchOn(iterator->value, [muted, interrupted](Ref<RemoteRealtimeAudioSource>& source) {
        source->sourceMutedChanged(muted, interrupted);
    }, [muted, interrupted](Ref<RemoteRealtimeVideoSource>& source) {
        source->sourceMutedChanged(muted, interrupted);
    }, [](std::nullptr_t) { });
}

void UserMediaCaptureManager::sourceSettingsChanged(RealtimeMediaSourceIdentifier identifier, RealtimeMediaSourceSettings&& settings)
{
    auto iterator = m_sources.find(identifier);
    if (iterator == m_sources.end())
        return;

    switchOn(iterator->value, [&](Ref<RemoteRealtimeAudioSource>& source) {
        source->setSettings(WTFMove(settings));
    }, [&](Ref<RemoteRealtimeVideoSource>& source) {
        source->setSettings(WTFMove(settings));
    }, [](std::nullptr_t) { });
}

void UserMediaCaptureManager::sourceConfigurationChanged(RealtimeMediaSourceIdentifier identifier, String&& persistentID, RealtimeMediaSourceSettings&& settings, RealtimeMediaSourceCapabilities&& capabilities)
{
    auto iterator = m_sources.find(identifier);
    if (iterator == m_sources.end())
        return;

    switchOn(iterator->value, [&](Ref<RemoteRealtimeAudioSource>& source) {
        source->configurationChanged(WTFMove(persistentID), WTFMove(settings), WTFMove(capabilities));
    }, [&](Ref<RemoteRealtimeVideoSource>& source) {
        source->configurationChanged(WTFMove(persistentID), WTFMove(settings), WTFMove(capabilities));
    }, [](std::nullptr_t) { });
}

void UserMediaCaptureManager::applyConstraintsSucceeded(RealtimeMediaSourceIdentifier identifier, RealtimeMediaSourceSettings&& settings)
{
    auto iterator = m_sources.find(identifier);
    if (iterator == m_sources.end())
        return;

    switchOn(iterator->value, [&](Ref<RemoteRealtimeAudioSource>& source) {
        source->applyConstraintsSucceeded(WTFMove(settings));
    }, [&](Ref<RemoteRealtimeVideoSource>& source) {
        source->applyConstraintsSucceeded(WTFMove(settings));
    }, [](std::nullptr_t) { });
}

void UserMediaCaptureManager::applyConstraintsFailed(RealtimeMediaSourceIdentifier identifier, WebCore::MediaConstraintType invalidConstraint, String&& message)
{
    auto iterator = m_sources.find(identifier);
    if (iterator == m_sources.end())
        return;

    switchOn(iterator->value, [&](Ref<RemoteRealtimeAudioSource>& source) {
        source->applyConstraintsFailed(invalidConstraint, WTFMove(message));
    }, [&](Ref<RemoteRealtimeVideoSource>& source) {
        source->applyConstraintsFailed(invalidConstraint, WTFMove(message));
    }, [](std::nullptr_t) { });
}

CaptureSourceOrError UserMediaCaptureManager::AudioFactory::createAudioCaptureSource(const CaptureDevice& device, MediaDeviceHashSalts&& hashSalts, const MediaConstraints* constraints, std::optional<PageIdentifier> pageIdentifier)
{
#if !ENABLE(GPU_PROCESS)
    if (m_shouldCaptureInGPUProcess)
        return CaptureSourceOrError { "Audio capture in GPUProcess is not implemented"_s };
#endif

#if PLATFORM(IOS_FAMILY) || ENABLE(ROUTING_ARBITRATION)
    // FIXME: Remove disabling of the audio session category management once we move all media playing to GPUProcess.
    if (m_shouldCaptureInGPUProcess)
        DeprecatedGlobalSettings::setShouldManageAudioSessionCategory(true);
#endif

    return RemoteRealtimeAudioSource::create(device, constraints, WTFMove(hashSalts), m_manager, m_shouldCaptureInGPUProcess, pageIdentifier);
}

void UserMediaCaptureManager::AudioFactory::setShouldCaptureInGPUProcess(bool value)
{
    m_shouldCaptureInGPUProcess = value;
}

void UserMediaCaptureManager::VideoFactory::setShouldCaptureInGPUProcess(bool value)
{
    m_shouldCaptureInGPUProcess = value;
}

CaptureSourceOrError UserMediaCaptureManager::VideoFactory::createVideoCaptureSource(const CaptureDevice& device, MediaDeviceHashSalts&& hashSalts, const MediaConstraints* constraints, std::optional<PageIdentifier> pageIdentifier)
{
#if !ENABLE(GPU_PROCESS)
    if (m_shouldCaptureInGPUProcess)
        return CaptureSourceOrError { "Video capture in GPUProcess is not implemented"_s };
#endif
    if (m_shouldCaptureInGPUProcess)
        m_manager->m_remoteCaptureSampleManager.setVideoFrameObjectHeapProxy(&WebProcess::singleton().ensureGPUProcessConnection().videoFrameObjectHeapProxy());

    return RemoteRealtimeVideoSource::create(device, constraints, WTFMove(hashSalts), m_manager, m_shouldCaptureInGPUProcess, pageIdentifier);
}

CaptureSourceOrError UserMediaCaptureManager::DisplayFactory::createDisplayCaptureSource(const CaptureDevice& device, MediaDeviceHashSalts&& hashSalts, const MediaConstraints* constraints, std::optional<PageIdentifier> pageIdentifier)
{
#if !ENABLE(GPU_PROCESS)
    if (m_shouldCaptureInGPUProcess)
        return CaptureSourceOrError { "Display capture in GPUProcess is not implemented"_s };
#endif
    if (m_shouldCaptureInGPUProcess) {
        Ref videoFrameObjectHeapProxy = WebProcess::singleton().ensureGPUProcessConnection().videoFrameObjectHeapProxy();
        m_manager->m_remoteCaptureSampleManager.setVideoFrameObjectHeapProxy(WTFMove(videoFrameObjectHeapProxy));
    }

    return RemoteRealtimeVideoSource::create(device, constraints, WTFMove(hashSalts), m_manager, m_shouldCaptureInGPUProcess, pageIdentifier);
}

void UserMediaCaptureManager::DisplayFactory::setShouldCaptureInGPUProcess(bool value)
{
    m_shouldCaptureInGPUProcess = value;
}


}

#endif // PLATFORM(COCOA) && ENABLE(MEDIA_STREAM)
