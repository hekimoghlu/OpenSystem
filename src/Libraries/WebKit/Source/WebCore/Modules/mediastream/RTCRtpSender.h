/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 29, 2022.
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

#if ENABLE(WEB_RTC)

#include "JSDOMPromiseDeferredForward.h"
#include "MediaStreamTrack.h"
#include "RTCDtlsTransport.h"
#include "RTCRtpSenderBackend.h"
#include "RTCRtpTransceiverDirection.h"
#include "RTCRtpTransform.h"
#include "ScriptWrappable.h"
#include <wtf/WeakPtr.h>

namespace WebCore {

class MediaStream;
class RTCDTMFSender;
class RTCPeerConnection;
struct RTCRtpCapabilities;

class RTCRtpSender final : public RefCounted<RTCRtpSender>
    , public ScriptWrappable
    , public CanMakeWeakPtr<RTCRtpSender>
#if !RELEASE_LOG_DISABLED
    , private LoggerHelper
#endif
    {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RTCRtpSender);
public:
    static Ref<RTCRtpSender> create(RTCPeerConnection&, Ref<MediaStreamTrack>&&, std::unique_ptr<RTCRtpSenderBackend>&&);
    static Ref<RTCRtpSender> create(RTCPeerConnection&, String&& trackKind, std::unique_ptr<RTCRtpSenderBackend>&&);
    virtual ~RTCRtpSender();

    static std::optional<RTCRtpCapabilities> getCapabilities(ScriptExecutionContext&, const String& kind);

    MediaStreamTrack* track() { return m_track.get(); }

    RTCDtlsTransport* transport() { return m_transport.get(); }
    void setTransport(RefPtr<RTCDtlsTransport>&& transport) { m_transport = WTFMove(transport); }

    const String& trackId() const { return m_trackId; }
    const String& trackKind() const { return m_trackKind; }

    ExceptionOr<void> setMediaStreamIds(const FixedVector<String>&);
    ExceptionOr<void> setStreams(const FixedVector<std::reference_wrapper<MediaStream>>&);

    bool isStopped() const { return !m_backend; }
    void stop();
    void setTrack(Ref<MediaStreamTrack>&&);
    void setTrackToNull();

    void replaceTrack(RefPtr<MediaStreamTrack>&&, Ref<DeferredPromise>&&);

    RTCRtpSendParameters getParameters();
    void setParameters(const RTCRtpSendParameters&, DOMPromiseDeferred<void>&&);

    RTCRtpSenderBackend* backend() { return m_backend.get(); }

    void getStats(Ref<DeferredPromise>&&);

    bool isCreatedBy(const RTCPeerConnection&) const;

    RTCDTMFSender* dtmf();
    std::optional<RTCRtpTransceiverDirection> currentTransceiverDirection() const;

    std::optional<RTCRtpTransform::Internal> transform();
    ExceptionOr<void> setTransform(std::unique_ptr<RTCRtpTransform>&&);

private:
    RTCRtpSender(RTCPeerConnection&, String&& trackKind, std::unique_ptr<RTCRtpSenderBackend>&&);

#if !RELEASE_LOG_DISABLED
    const Logger& logger() const final { return m_logger.get(); }
    uint64_t logIdentifier() const final { return m_logIdentifier; }
    ASCIILiteral logClassName() const final { return "RTCRtpSender"_s; }
    WTFLogChannel& logChannel() const final;
#endif

    RefPtr<MediaStreamTrack> m_track;
    RefPtr<RTCDtlsTransport> m_transport;
    String m_trackId;
    String m_trackKind;
    std::unique_ptr<RTCRtpSenderBackend> m_backend;
    WeakPtr<RTCPeerConnection, WeakPtrImplWithEventTargetData> m_connection;
    RefPtr<RTCDTMFSender> m_dtmfSender;
    std::unique_ptr<RTCRtpTransform> m_transform;
#if !RELEASE_LOG_DISABLED
    Ref<const Logger> m_logger;
    uint64_t m_logIdentifier { 0 };
#endif
};

} // namespace WebCore

#endif // ENABLE(WEB_RTC)
