/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 16, 2025.
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

#include "MediaStream.h"
#include "RTCDtlsTransport.h"
#include "RTCRtpReceiverBackend.h"
#include "RTCRtpSynchronizationSource.h"
#include "RTCRtpTransform.h"
#include "ScriptWrappable.h"

namespace WebCore {

class DeferredPromise;
class PeerConnectionBackend;
struct RTCRtpCapabilities;

class RTCRtpReceiver final : public RefCounted<RTCRtpReceiver>
    , public ScriptWrappable
#if !RELEASE_LOG_DISABLED
    , private LoggerHelper
#endif
    {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RTCRtpReceiver);
public:
    static Ref<RTCRtpReceiver> create(PeerConnectionBackend& connection, Ref<MediaStreamTrack>&& track, std::unique_ptr<RTCRtpReceiverBackend>&& backend)
    {
        return adoptRef(*new RTCRtpReceiver(connection, WTFMove(track), WTFMove(backend)));
    }
    virtual ~RTCRtpReceiver();

    static std::optional<RTCRtpCapabilities> getCapabilities(ScriptExecutionContext&, const String& kind);

    void stop();

    void setBackend(std::unique_ptr<RTCRtpReceiverBackend>&& backend) { m_backend = WTFMove(backend); }
    RTCRtpParameters getParameters() { return m_backend ? m_backend->getParameters() : RTCRtpParameters(); }
    Vector<RTCRtpContributingSource> getContributingSources() const { return m_backend ? m_backend->getContributingSources() : Vector<RTCRtpContributingSource> { }; }
    Vector<RTCRtpSynchronizationSource> getSynchronizationSources() const { return m_backend ? m_backend->getSynchronizationSources() : Vector<RTCRtpSynchronizationSource> { }; }

    MediaStreamTrack& track() { return m_track.get(); }

    RTCDtlsTransport* transport() { return m_transport.get(); }
    void setTransport(RefPtr<RTCDtlsTransport>&& transport) { m_transport = WTFMove(transport); }

    RTCRtpReceiverBackend* backend() { return m_backend.get(); }
    void getStats(Ref<DeferredPromise>&&);

    std::optional<RTCRtpTransform::Internal> transform();
    ExceptionOr<void> setTransform(std::unique_ptr<RTCRtpTransform>&&);

    const Vector<WeakPtr<MediaStream>>& associatedStreams() const { return m_associatedStreams; }
    void setAssociatedStreams(Vector<WeakPtr<MediaStream>>&& streams) { m_associatedStreams = WTFMove(streams); }
private:
    RTCRtpReceiver(PeerConnectionBackend&, Ref<MediaStreamTrack>&&, std::unique_ptr<RTCRtpReceiverBackend>&&);

#if !RELEASE_LOG_DISABLED
    const Logger& logger() const final { return m_logger.get(); }
    uint64_t logIdentifier() const final { return m_logIdentifier; }
    ASCIILiteral logClassName() const final { return "RTCRtpReceiver"_s; }
    WTFLogChannel& logChannel() const final;
#endif

    Ref<MediaStreamTrack> m_track;
    RefPtr<RTCDtlsTransport> m_transport;
    std::unique_ptr<RTCRtpReceiverBackend> m_backend;
    WeakPtr<PeerConnectionBackend> m_connection;
    std::unique_ptr<RTCRtpTransform> m_transform;
    Vector<WeakPtr<MediaStream>> m_associatedStreams;
#if !RELEASE_LOG_DISABLED
    Ref<const Logger> m_logger;
    uint64_t m_logIdentifier { 0 };
#endif
};

} // namespace WebCore

#endif // ENABLE(WEB_RTC)
