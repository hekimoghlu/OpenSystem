/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 20, 2025.
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

#if ENABLE(GPU_PROCESS)

#include "AuxiliaryProcess.h"
#include "GPUProcessPreferences.h"
#include "SandboxExtension.h"
#include "WebPageProxyIdentifier.h"
#include <WebCore/IntDegrees.h>
#include <WebCore/MediaPlayerIdentifier.h>
#include <WebCore/PageIdentifier.h>
#include <WebCore/ProcessIdentity.h>
#include <WebCore/ShareableBitmap.h>
#include <WebCore/SnapshotIdentifier.h>
#include <WebCore/Timer.h>
#include <pal/SessionID.h>
#include <wtf/Function.h>
#include <wtf/MemoryPressureHandler.h>
#include <wtf/MonotonicTime.h>
#include <wtf/WeakPtr.h>

#if PLATFORM(MAC)
#include <CoreGraphics/CGDisplayConfiguration.h>
#endif

#if USE(GRAPHICS_LAYER_WC)
#include "WCSharedSceneContextHolder.h"
#endif

#if HAVE(SCREEN_CAPTURE_KIT)
#include <WebCore/DisplayCapturePromptType.h>
#endif

#if PLATFORM(VISION) && ENABLE(MODEL_PROCESS)
namespace IPC {
class SharedFileHandle;
}
#endif

namespace WebCore {
class CaptureDevice;
class NowPlayingManager;
class SecurityOriginData;

struct MockMediaDevice;
struct ScreenProperties;

enum class VideoFrameRotation : uint16_t;
}

namespace WebKit {

class GPUConnectionToWebProcess;
class RemoteAudioSessionProxyManager;
struct CoreIPCAuditToken;
struct GPUProcessConnectionParameters;
struct GPUProcessCreationParameters;
struct GPUProcessSessionParameters;
struct SharedPreferencesForWebProcess;

class GPUProcess final : public AuxiliaryProcess, public ThreadSafeRefCounted<GPUProcess> {
    WTF_MAKE_NONCOPYABLE(GPUProcess);
    WTF_MAKE_FAST_ALLOCATED;
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(GPUProcess);
public:
    GPUProcess();
    ~GPUProcess();

    static GPUProcess& singleton();
    static constexpr WTF::AuxiliaryProcessType processType = WTF::AuxiliaryProcessType::GPU;

    void ref() const final { ThreadSafeRefCounted::ref(); }
    void deref() const final { ThreadSafeRefCounted::deref(); }

    void removeGPUConnectionToWebProcess(GPUConnectionToWebProcess&);

    void prepareToSuspend(bool isSuspensionImminent, MonotonicTime estimatedSuspendTime, CompletionHandler<void()>&&);
    void processDidResume();
    void resume();

    void connectionToWebProcessClosed(IPC::Connection&);

    GPUConnectionToWebProcess* webProcessConnection(WebCore::ProcessIdentifier) const;

    const String& mediaCacheDirectory(PAL::SessionID) const;
#if ENABLE(LEGACY_ENCRYPTED_MEDIA) || ENABLE(ENCRYPTED_MEDIA)
    const String& mediaKeysStorageDirectory(PAL::SessionID) const;
#endif

#if ENABLE(GPU_PROCESS) && USE(AUDIO_SESSION)
    RemoteAudioSessionProxyManager& audioSessionManager() const;
    Ref<RemoteAudioSessionProxyManager> protectedAudioSessionManager() const;
#endif

    WebCore::NowPlayingManager& nowPlayingManager();

#if ENABLE(MEDIA_STREAM) && PLATFORM(COCOA)
    WorkQueue& videoMediaStreamTrackRendererQueue();
    void ensureAVCaptureServerConnection();
#endif

#if USE(LIBWEBRTC) && PLATFORM(COCOA)
    WorkQueue& libWebRTCCodecsQueue();
#endif

#if USE(GRAPHICS_LAYER_WC)
    WCSharedSceneContextHolder& sharedSceneContext() { return m_sharedSceneContext; }
#endif

    void tryExitIfUnusedAndUnderMemoryPressure();

    const String& applicationVisibleName() const { return m_applicationVisibleName; }

    void webProcessConnectionCountForTesting(CompletionHandler<void(uint64_t)>&&);

#if USE(EXTENSIONKIT)
    void resolveBookmarkDataForCacheDirectory(std::span<const uint8_t> bookmarkData);
#endif

#if PLATFORM(COCOA) && ENABLE(MEDIA_STREAM)
    void processIsStartingToCaptureAudio(GPUConnectionToWebProcess&);
#endif

#if ENABLE(VIDEO)
    void requestBitmapImageForCurrentTime(WebCore::ProcessIdentifier, WebCore::MediaPlayerIdentifier, CompletionHandler<void(std::optional<WebCore::ShareableBitmap::Handle>&&)>&&);
#endif
#if ENABLE(WEBXR)
    std::optional<WebCore::ProcessIdentity> immersiveModeProcessIdentity() const;
#endif

#if HAVE(AUDIT_TOKEN)
    void setPresentingApplicationAuditToken(WebCore::ProcessIdentifier, WebCore::PageIdentifier, std::optional<CoreIPCAuditToken>&&);
#endif

#if PLATFORM(COCOA)
    void didDrawRemoteToPDF(WebCore::PageIdentifier, RefPtr<WebCore::SharedBuffer>&&, WebCore::SnapshotIdentifier);
#endif

#if PLATFORM(VISION) && ENABLE(MODEL_PROCESS)
    void requestSharedSimulationConnection(CoreIPCAuditToken&&, CompletionHandler<void(std::optional<IPC::SharedFileHandle>)>&&);
#endif

private:
    void lowMemoryHandler(Critical, Synchronous);

    // AuxiliaryProcess
    void initializeProcess(const AuxiliaryProcessInitializationParameters&) override;
    void initializeProcessName(const AuxiliaryProcessInitializationParameters&) override;
    void initializeSandbox(const AuxiliaryProcessInitializationParameters&, SandboxInitializationParameters&) override;
    bool shouldTerminate() override;

    void tryExitIfUnused();
    bool canExitUnderMemoryPressure() const;

    // IPC::Connection::Client
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) override;

    // Message Handlers
    void initializeGPUProcess(GPUProcessCreationParameters&&, CompletionHandler<void()>&&);
    void platformInitializeGPUProcess(GPUProcessCreationParameters&);
    void updateGPUProcessPreferences(GPUProcessPreferences&&);
    void createGPUConnectionToWebProcess(WebCore::ProcessIdentifier, PAL::SessionID, IPC::Connection::Handle&&, GPUProcessConnectionParameters&&, CompletionHandler<void()>&&);
    void sharedPreferencesForWebProcessDidChange(WebCore::ProcessIdentifier, SharedPreferencesForWebProcess&&, CompletionHandler<void()>&&);
    void addSession(PAL::SessionID, GPUProcessSessionParameters&&);
    void removeSession(PAL::SessionID);
    void updateSandboxAccess(const Vector<SandboxExtension::Handle>&);

    bool updatePreference(std::optional<bool>& oldPreference, std::optional<bool>& newPreference);
    void userPreferredLanguagesChanged(Vector<String>&&);

#if ENABLE(MEDIA_STREAM)
    void setMockCaptureDevicesEnabled(bool);
    void setUseSCContentSharingPicker(bool);
    void enableMicrophoneMuteStatusAPI();
    void setOrientationForMediaCapture(WebCore::IntDegrees);
    void rotationAngleForCaptureDeviceChanged(const String&, WebCore::VideoFrameRotation);
    void updateCaptureAccess(bool allowAudioCapture, bool allowVideoCapture, bool allowDisplayCapture, WebCore::ProcessIdentifier, CompletionHandler<void()>&&);
    void updateCaptureOrigin(const WebCore::SecurityOriginData&, WebCore::ProcessIdentifier);
    void addMockMediaDevice(const WebCore::MockMediaDevice&);
    void clearMockMediaDevices();
    void removeMockMediaDevice(const String&);
    void setMockMediaDeviceIsEphemeral(const String&, bool);
    void resetMockMediaDevices();
    void setMockCaptureDevicesInterrupted(bool isCameraInterrupted, bool isMicrophoneInterrupted);
    void triggerMockCaptureConfigurationChange(bool forMicrophone, bool forDisplay);
    void setShouldListenToVoiceActivity(bool);
#endif
#if HAVE(SCREEN_CAPTURE_KIT)
    void promptForGetDisplayMedia(WebCore::DisplayCapturePromptType, CompletionHandler<void(std::optional<WebCore::CaptureDevice>)>&&);
    void cancelGetDisplayMediaPrompt();
#endif
#if PLATFORM(MAC)
    void setScreenProperties(const WebCore::ScreenProperties&);
    void updateProcessName();
#endif

#if USE(OS_STATE)
    RetainPtr<NSDictionary> additionalStateForDiagnosticReport() const final;
#endif

#if ENABLE(CFPREFS_DIRECT_MODE)
    void dispatchSimulatedNotificationsForPreferenceChange(const String& key) final;
#endif

#if PLATFORM(MAC)
    void openDirectoryCacheInvalidated(SandboxExtension::Handle&&);
#endif

#if HAVE(POWERLOG_TASK_MODE_QUERY)
    void enablePowerLogging(SandboxExtension::Handle&&);
#endif
#if ENABLE(WEBXR)
    void webXRPromptAccepted(std::optional<WebCore::ProcessIdentity>, CompletionHandler<void(bool)>&&);
#endif

    // Connections to WebProcesses.
    HashMap<WebCore::ProcessIdentifier, Ref<GPUConnectionToWebProcess>> m_webProcessConnections;
    MonotonicTime m_creationTime { MonotonicTime::now() };

    GPUProcessPreferences m_preferences;

#if ENABLE(MEDIA_STREAM)
    struct MediaCaptureAccess {
        bool allowAudioCapture { false };
        bool allowVideoCapture { false };
        bool allowDisplayCapture { false };
    };
    HashMap<WebCore::ProcessIdentifier, MediaCaptureAccess> m_mediaCaptureAccessMap;
#if ENABLE(MEDIA_STREAM) && PLATFORM(COCOA)
    RefPtr<WorkQueue> m_videoMediaStreamTrackRendererQueue;
#endif
    WebCore::IntDegrees m_orientation { 0 };
#endif
#if USE(LIBWEBRTC) && PLATFORM(COCOA)
    RefPtr<WorkQueue> m_libWebRTCCodecsQueue;
#endif

#if USE(GRAPHICS_LAYER_WC)
    WCSharedSceneContextHolder m_sharedSceneContext;
#endif

    struct GPUSession {
        String mediaCacheDirectory;
#if ENABLE(LEGACY_ENCRYPTED_MEDIA) || ENABLE(ENCRYPTED_MEDIA)
        String mediaKeysStorageDirectory;
#endif
    };
    HashMap<PAL::SessionID, GPUSession> m_sessions;
    WebCore::Timer m_idleExitTimer;
    std::unique_ptr<WebCore::NowPlayingManager> m_nowPlayingManager;
    String m_applicationVisibleName;
#if PLATFORM(MAC)
    String m_uiProcessName;
#endif
#if ENABLE(GPU_PROCESS) && USE(AUDIO_SESSION)
    mutable RefPtr<RemoteAudioSessionProxyManager> m_audioSessionManager;
#endif
#if ENABLE(WEBXR)
    std::optional<WebCore::ProcessIdentity> m_processIdentity;
#endif
#if ENABLE(VP9) && PLATFORM(COCOA)
    bool m_haveEnabledVP9Decoder { false };
    bool m_haveEnabledSWVP9Decoder { false };
#endif

};

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS)
