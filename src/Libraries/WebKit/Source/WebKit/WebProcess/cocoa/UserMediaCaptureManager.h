/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 6, 2024.
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

#include "RemoteCaptureSampleManager.h"
#include "WebProcessSupplement.h"
#include <WebCore/DisplayCaptureManager.h>
#include <WebCore/RealtimeMediaSource.h>
#include <WebCore/RealtimeMediaSourceFactory.h>
#include <WebCore/RealtimeMediaSourceIdentifier.h>
#include <WebCore/SharedMemory.h>
#include <wtf/CheckedPtr.h>
#include <wtf/HashMap.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {
class CAAudioStreamDescription;
}

namespace WebKit {

class RemoteRealtimeAudioSource;
class RemoteRealtimeVideoSource;
class WebProcess;

class UserMediaCaptureManager final : public WebProcessSupplement, public IPC::MessageReceiver, public CanMakeCheckedPtr<UserMediaCaptureManager> {
    WTF_MAKE_TZONE_ALLOCATED(UserMediaCaptureManager);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(UserMediaCaptureManager);
public:
    explicit UserMediaCaptureManager(WebProcess&);
    ~UserMediaCaptureManager();

    void ref() const final;
    void deref() const final;

    static ASCIILiteral supplementName();

    void didReceiveMessageFromGPUProcess(IPC::Connection& connection, IPC::Decoder& decoder) { didReceiveMessage(connection, decoder); }
    void setupCaptureProcesses(bool shouldCaptureAudioInUIProcess, bool shouldCaptureAudioInGPUProcess, bool shouldCaptureVideoInUIProcess, bool shouldCaptureVideoInGPUProcess, bool shouldCaptureDisplayInUIProcess, bool shouldCaptureDisplayInGPUProcess, bool shouldUseGPUProcessRemoteFrames);

    void addSource(Ref<RemoteRealtimeAudioSource>&&);
    void addSource(Ref<RemoteRealtimeVideoSource>&&);
    void removeSource(WebCore::RealtimeMediaSourceIdentifier);

    RemoteCaptureSampleManager& remoteCaptureSampleManager() { return m_remoteCaptureSampleManager; }
    bool shouldUseGPUProcessRemoteFrames() const { return m_shouldUseGPUProcessRemoteFrames; }

private:
    // WebCore::RealtimeMediaSource factories
    class AudioFactory : public WebCore::AudioCaptureFactory {
    public:
        explicit AudioFactory(UserMediaCaptureManager& manager) : m_manager(manager) { }
        void setShouldCaptureInGPUProcess(bool);

    private:
        WebCore::CaptureSourceOrError createAudioCaptureSource(const WebCore::CaptureDevice&, WebCore::MediaDeviceHashSalts&&, const WebCore::MediaConstraints*, std::optional<WebCore::PageIdentifier>) final;
        WebCore::CaptureDeviceManager& audioCaptureDeviceManager() final { return m_manager->m_noOpCaptureDeviceManager; }
        const Vector<WebCore::CaptureDevice>& speakerDevices() const final { return m_speakerDevices; }

        CheckedRef<UserMediaCaptureManager> m_manager;
        bool m_shouldCaptureInGPUProcess { false };
        Vector<WebCore::CaptureDevice> m_speakerDevices;
    };
    class VideoFactory : public WebCore::VideoCaptureFactory {
    public:
        explicit VideoFactory(UserMediaCaptureManager& manager) : m_manager(manager) { }
        void setShouldCaptureInGPUProcess(bool);

    private:
        WebCore::CaptureSourceOrError createVideoCaptureSource(const WebCore::CaptureDevice&, WebCore::MediaDeviceHashSalts&&, const WebCore::MediaConstraints*, std::optional<WebCore::PageIdentifier>) final;
        WebCore::CaptureDeviceManager& videoCaptureDeviceManager() final { return m_manager->m_noOpCaptureDeviceManager; }

        CheckedRef<UserMediaCaptureManager> m_manager;
        bool m_shouldCaptureInGPUProcess { false };
    };
    class DisplayFactory : public WebCore::DisplayCaptureFactory {
    public:
        explicit DisplayFactory(UserMediaCaptureManager& manager) : m_manager(manager) { }
        void setShouldCaptureInGPUProcess(bool);

    private:
        WebCore::CaptureSourceOrError createDisplayCaptureSource(const WebCore::CaptureDevice&, WebCore::MediaDeviceHashSalts&&, const WebCore::MediaConstraints*, std::optional<WebCore::PageIdentifier>) final;
        WebCore::DisplayCaptureManager& displayCaptureDeviceManager() final { return m_manager->m_noOpCaptureDeviceManager; }

        CheckedRef<UserMediaCaptureManager> m_manager;
        bool m_shouldCaptureInGPUProcess { false };
    };

    class NoOpCaptureDeviceManager : public WebCore::DisplayCaptureManager {
    public:
        NoOpCaptureDeviceManager() = default;

    private:
        const Vector<WebCore::CaptureDevice>& captureDevices() final
        {
            ASSERT_NOT_REACHED();
            return m_emptyDevices;
        }
        Vector<WebCore::CaptureDevice> m_emptyDevices;
    };

    // IPC::MessageReceiver
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) final;

    // Messages::UserMediaCaptureManager
    void sourceStopped(WebCore::RealtimeMediaSourceIdentifier, bool didFail);
    void sourceMutedChanged(WebCore::RealtimeMediaSourceIdentifier, bool muted, bool interrupted);

    void sourceSettingsChanged(WebCore::RealtimeMediaSourceIdentifier, WebCore::RealtimeMediaSourceSettings&&);
    void sourceConfigurationChanged(WebCore::RealtimeMediaSourceIdentifier, String&&, WebCore::RealtimeMediaSourceSettings&&, WebCore::RealtimeMediaSourceCapabilities&&);

    void applyConstraintsSucceeded(WebCore::RealtimeMediaSourceIdentifier, WebCore::RealtimeMediaSourceSettings&&);
    void applyConstraintsFailed(WebCore::RealtimeMediaSourceIdentifier, WebCore::MediaConstraintType, String&&);

    CheckedRef<WebProcess> m_process;
    using Source = std::variant<std::nullptr_t, Ref<RemoteRealtimeAudioSource>, Ref<RemoteRealtimeVideoSource>>;
    HashMap<WebCore::RealtimeMediaSourceIdentifier, Source> m_sources;
    NoOpCaptureDeviceManager m_noOpCaptureDeviceManager;
    AudioFactory m_audioFactory;
    VideoFactory m_videoFactory;
    DisplayFactory m_displayFactory;
    RemoteCaptureSampleManager m_remoteCaptureSampleManager;
    bool m_shouldUseGPUProcessRemoteFrames { false };
};

} // namespace WebKit

#endif
