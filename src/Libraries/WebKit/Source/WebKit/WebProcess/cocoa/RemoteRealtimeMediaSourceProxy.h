/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 26, 2023.
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

#include <WebCore/RealtimeMediaSource.h>
#include <wtf/Deque.h>

namespace IPC {
class Connection;
}

namespace WebCore {
class CAAudioStreamDescription;
class ImageTransferSessionVT;
struct MediaConstraints;
enum class MediaAccessDenialReason : uint8_t;
}

namespace WebKit {

class RemoteRealtimeMediaSourceProxy {
public:
    RemoteRealtimeMediaSourceProxy(WebCore::RealtimeMediaSourceIdentifier, const WebCore::CaptureDevice&, bool shouldCaptureInGPUProcess, const WebCore::MediaConstraints*);
    ~RemoteRealtimeMediaSourceProxy();

    RemoteRealtimeMediaSourceProxy(RemoteRealtimeMediaSourceProxy&&) = default;
    RemoteRealtimeMediaSourceProxy& operator=(RemoteRealtimeMediaSourceProxy&&) = default;

    IPC::Connection& connection() { return m_connection.get(); }
    WebCore::RealtimeMediaSourceIdentifier identifier() const { return m_identifier; }
    WebCore::CaptureDevice::DeviceType deviceType() const { return m_device.type(); }
    const WebCore::CaptureDevice& device() const { return m_device; }
    bool shouldCaptureInGPUProcess() const { return m_shouldCaptureInGPUProcess; }

    using CreateCallback = CompletionHandler<void(WebCore::CaptureSourceError&&, WebCore::RealtimeMediaSourceSettings&&, WebCore::RealtimeMediaSourceCapabilities&&)>;
    void createRemoteMediaSource(const WebCore::MediaDeviceHashSalts&, WebCore::PageIdentifier, CreateCallback&&, bool shouldUseRemoteFrame = false);

    RemoteRealtimeMediaSourceProxy clone();
    void createRemoteCloneSource(WebCore::RealtimeMediaSourceIdentifier, WebCore::PageIdentifier);

    void applyConstraintsSucceeded();
    void applyConstraintsFailed(WebCore::MediaConstraintType, String&& errorMessage);
    void failApplyConstraintCallbacks(const String& errorMessage);

    bool isEnded() const { return m_isEnded; }
    void end();
    void startProducingData(WebCore::PageIdentifier);
    void stopProducingData();
    void endProducingData();
    void applyConstraints(const WebCore::MediaConstraints&, WebCore::RealtimeMediaSource::ApplyConstraintsHandler&&);

    Ref<WebCore::RealtimeMediaSource::TakePhotoNativePromise> takePhoto(WebCore::PhotoSettings&&);
    Ref<WebCore::RealtimeMediaSource::PhotoCapabilitiesNativePromise> getPhotoCapabilities();
    Ref<WebCore::RealtimeMediaSource::PhotoSettingsNativePromise> getPhotoSettings();

    void whenReady(CompletionHandler<void(WebCore::CaptureSourceError&&)>&&);
    void setAsReady();
    void resetReady() { m_isReady = false; }
    bool isReady() const { return m_isReady; }

    void didFail(WebCore::CaptureSourceError&&);

    bool interrupted() const { return m_interrupted; }
    void setInterrupted(bool interrupted) { m_interrupted = interrupted; }

    void updateConnection();

    bool isPowerEfficient() const;

private:
    struct PromiseConverter;

    WebCore::RealtimeMediaSourceIdentifier m_identifier;
    Ref<IPC::Connection> m_connection;
    WebCore::CaptureDevice m_device;
    bool m_shouldCaptureInGPUProcess { false };

    WebCore::MediaConstraints m_constraints;
    Deque<std::pair<WebCore::RealtimeMediaSource::ApplyConstraintsHandler, WebCore::MediaConstraints>> m_pendingApplyConstraintsRequests;
    bool m_isReady { false };
    CompletionHandler<void(WebCore::CaptureSourceError&&)> m_callback;
    WebCore::CaptureSourceError m_failureReason;
    bool m_interrupted { false };
    bool m_isEnded { false };
};

} // namespace WebKit

#endif
