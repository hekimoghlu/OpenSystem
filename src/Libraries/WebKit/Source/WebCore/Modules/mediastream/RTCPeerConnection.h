/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 17, 2025.
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

#include "ActiveDOMObject.h"
#include "EventTarget.h"
#include "MediaEndpointConfiguration.h"
#include "MediaStream.h"
#include "PeerConnectionBackend.h"
#include "RTCConfiguration.h"
#include "RTCDataChannel.h"
#include "RTCIceConnectionState.h"
#include "RTCIceGatheringState.h"
#include "RTCLocalSessionDescriptionInit.h"
#include "RTCPeerConnectionState.h"
#include "RTCRtpEncodingParameters.h"
#include "RTCRtpTransceiver.h"
#include "RTCSessionDescriptionInit.h"
#include "RTCSignalingState.h"
#include <JavaScriptCore/Forward.h>
#include <wtf/LoggerHelper.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class DOMPromiseDeferredBase;
class MediaStream;
class MediaStreamTrack;
class RTCController;
class RTCDtlsTransport;
class RTCDtlsTransportBackend;
class RTCIceCandidate;
class RTCIceTransportBackend;
class RTCPeerConnectionErrorCallback;
class RTCSctpTransport;
class RTCSessionDescription;
class RTCStatsCallback;

struct RTCAnswerOptions;
struct RTCOfferOptions;
struct RTCRtpParameters;
struct RTCIceCandidateInit;
struct RTCSessionDescriptionInit;

struct RTCRtpTransceiverInit {
    RTCRtpTransceiverDirection direction { RTCRtpTransceiverDirection::Sendrecv };
    Vector<Ref<MediaStream>> streams;
    Vector<RTCRtpEncodingParameters> sendEncodings;
};

class RTCPeerConnection final
    : public RefCounted<RTCPeerConnection>
    , public EventTarget
    , public ActiveDOMObject
#if !RELEASE_LOG_DISABLED
    , private LoggerHelper
#endif
{
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED_EXPORT(RTCPeerConnection, WEBCORE_EXPORT);
public:
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    static ExceptionOr<Ref<RTCPeerConnection>> create(Document&, RTCConfiguration&&);
    WEBCORE_EXPORT virtual ~RTCPeerConnection();

    using DataChannelInit = RTCDataChannelInit;

    struct CertificateParameters {
        String name;
        String hash;
        String namedCurve;
        std::optional<uint32_t> modulusLength;
        RefPtr<Uint8Array> publicExponent;
        std::optional<double> expires;
    };

    using AlgorithmIdentifier = std::variant<JSC::Strong<JSC::JSObject>, String>;
    static void generateCertificate(JSC::JSGlobalObject&, AlgorithmIdentifier&&, DOMPromiseDeferred<IDLInterface<RTCCertificate>>&&);

    // 4.3.2 RTCPeerConnection Interface
    void createOffer(RTCOfferOptions&&, Ref<DeferredPromise>&&);
    void createAnswer(RTCAnswerOptions&&, Ref<DeferredPromise>&&);

    void setLocalDescription(std::optional<RTCLocalSessionDescriptionInit>&&, Ref<DeferredPromise>&&);
    RefPtr<RTCSessionDescription> localDescription() const { return m_pendingLocalDescription ? m_pendingLocalDescription.get() : m_currentLocalDescription.get(); }
    RefPtr<RTCSessionDescription> currentLocalDescription() const { return m_currentLocalDescription.get(); }
    RefPtr<RTCSessionDescription> pendingLocalDescription() const { return m_pendingLocalDescription.get(); }

    void setRemoteDescription(RTCSessionDescriptionInit&&, Ref<DeferredPromise>&&);
    RTCSessionDescription* remoteDescription() const { return m_pendingRemoteDescription ? m_pendingRemoteDescription.get() : m_currentRemoteDescription.get(); }
    RTCSessionDescription* currentRemoteDescription() const { return m_currentRemoteDescription.get(); }
    RTCSessionDescription* pendingRemoteDescription() const { return m_pendingRemoteDescription.get(); }

    using Candidate = std::optional<std::variant<RTCIceCandidateInit, RefPtr<RTCIceCandidate>>>;
    void addIceCandidate(Candidate&&, Ref<DeferredPromise>&&);

    RTCSignalingState signalingState() const { return m_signalingState; }
    RTCIceGatheringState iceGatheringState() const { return m_iceGatheringState; }
    RTCIceConnectionState iceConnectionState() const { return m_iceConnectionState; }
    RTCPeerConnectionState connectionState() const { return m_connectionState; }
    std::optional<bool> canTrickleIceCandidates() const;

    void restartIce() { protectedBackend()->restartIce(); }
    const RTCConfiguration& getConfiguration() const { return m_configuration; }
    ExceptionOr<void> setConfiguration(RTCConfiguration&&);
    void close();

    bool isClosed() const { return m_connectionState == RTCPeerConnectionState::Closed; }
    bool isStopped() const { return m_isStopped; }

    void addInternalTransceiver(Ref<RTCRtpTransceiver>&&);

    // 5.1 RTCPeerConnection extensions
    Vector<std::reference_wrapper<RTCRtpSender>> getSenders() const;
    Vector<std::reference_wrapper<RTCRtpReceiver>> getReceivers() const;
    const Vector<RefPtr<RTCRtpTransceiver>>& getTransceivers() const;

    const Vector<RefPtr<RTCRtpTransceiver>>& currentTransceivers() const { return m_transceiverSet.list(); }

    ExceptionOr<Ref<RTCRtpSender>> addTrack(Ref<MediaStreamTrack>&&, const FixedVector<std::reference_wrapper<MediaStream>>&);
    ExceptionOr<void> removeTrack(RTCRtpSender&);

    using AddTransceiverTrackOrKind = std::variant<RefPtr<MediaStreamTrack>, String>;
    ExceptionOr<Ref<RTCRtpTransceiver>> addTransceiver(AddTransceiverTrackOrKind&&, RTCRtpTransceiverInit&&);

    // 6.1 Peer-to-peer data API
    ExceptionOr<Ref<RTCDataChannel>> createDataChannel(String&&, RTCDataChannelInit&&);

    // 8.2 Statistics API
    void getStats(MediaStreamTrack*, Ref<DeferredPromise>&&);
    // Used for testing
    WEBCORE_EXPORT void gatherDecoderImplementationName(Function<void(String&&)>&&);

    // EventTarget
    enum EventTargetInterfaceType eventTargetInterface() const final { return EventTargetInterfaceType::RTCPeerConnection; }
    ScriptExecutionContext* scriptExecutionContext() const final { return ActiveDOMObject::scriptExecutionContext(); }

    // Used for testing with a mock
    WEBCORE_EXPORT void emulatePlatformEvent(const String& action);

    // API used by PeerConnectionBackend and relatives
    void updateIceGatheringState(RTCIceGatheringState);
    void updateIceConnectionState(RTCIceConnectionState);
    void updateConnectionState();

    void updateNegotiationNeededFlag(std::optional<uint32_t>);

    void scheduleEvent(Ref<Event>&&);

    void disableICECandidateFiltering() { protectedBackend()->disableICECandidateFiltering(); }
    void enableICECandidateFiltering() { protectedBackend()->enableICECandidateFiltering(); }

    void clearController() { m_controller = nullptr; }

    Document* document();

    void updateDescriptions(PeerConnectionBackend::DescriptionStates&&);
    void updateTransceiversAfterSuccessfulLocalDescription();
    void updateTransceiversAfterSuccessfulRemoteDescription();
    void updateSctpBackend(std::unique_ptr<RTCSctpTransportBackend>&&, std::optional<double>);

    void processIceTransportStateChange(RTCIceTransport&);
    void processIceTransportChanges();

    RTCSctpTransport* sctp() { return m_sctpTransport.get(); }

    // EventTarget implementation.
    void dispatchEvent(Event&) final;

    void dispatchDataChannelEvent(UniqueRef<RTCDataChannelHandler>&&, String&& label, RTCDataChannelInit&&);

#if !RELEASE_LOG_DISABLED
    const Logger& logger() const final { return m_logger.get(); }
    uint64_t logIdentifier() const final { return m_logIdentifier; }
    ASCIILiteral logClassName() const final { return "RTCPeerConnection"_s; }
    WTFLogChannel& logChannel() const final;
#endif

    void startGatheringStatLogs(Function<void(String&&)>&&);
    void stopGatheringStatLogs();

private:
    RTCPeerConnection(Document&);

    ExceptionOr<void> initializeConfiguration(RTCConfiguration&&);

    ExceptionOr<Ref<RTCRtpTransceiver>> addReceiveOnlyTransceiver(String&&);

    void registerToController(RTCController&);
    void unregisterFromController();

    friend class Internals;
    void applyRotationForOutgoingVideoSources() { protectedBackend()->applyRotationForOutgoingVideoSources(); }

    // EventTarget implementation.
    void refEventTarget() final { ref(); }
    void derefEventTarget() final { deref(); }

    // ActiveDOMObject
    WEBCORE_EXPORT void stop() final;
    void suspend(ReasonForSuspension) final;
    void resume() final;
    bool virtualHasPendingActivity() const final;

    bool updateIceConnectionStateFromIceTransports();
    RTCIceConnectionState computeIceConnectionStateFromIceTransports();
    RTCPeerConnectionState computeConnectionState();

    bool doClose();
    void doStop();

    void getStats(RTCRtpSender& sender, Ref<DeferredPromise>&& promise) { protectedBackend()->getStats(sender, WTFMove(promise)); }

    ExceptionOr<Vector<MediaEndpointConfiguration::CertificatePEM>> certificatesFromConfiguration(const RTCConfiguration&);
    void chainOperation(Ref<DeferredPromise>&&, Function<void(Ref<DeferredPromise>&&)>&&);
    friend class RTCRtpSender;

    ExceptionOr<Vector<MediaEndpointConfiguration::IceServerInfo>> iceServersFromConfiguration(RTCConfiguration& newConfiguration, const RTCConfiguration* existingConfiguration, bool isLocalDescriptionSet);

    Ref<RTCIceTransport> getOrCreateIceTransport(UniqueRef<RTCIceTransportBackend>&&);
    RefPtr<RTCDtlsTransport> getOrCreateDtlsTransport(std::unique_ptr<RTCDtlsTransportBackend>&&);
    void updateTransceiverTransports();

    void setSignalingState(RTCSignalingState);

    WEBCORE_EXPORT RefPtr<PeerConnectionBackend> protectedBackend() const;

    bool m_isStopped { false };
    RTCSignalingState m_signalingState { RTCSignalingState::Stable };
    RTCIceGatheringState m_iceGatheringState { RTCIceGatheringState::New };
    RTCIceConnectionState m_iceConnectionState { RTCIceConnectionState::New };
    RTCPeerConnectionState m_connectionState { RTCPeerConnectionState::New };

#if !RELEASE_LOG_DISABLED
    Ref<const Logger> m_logger;
    const uint64_t m_logIdentifier;
#endif

    RtpTransceiverSet m_transceiverSet;

    std::unique_ptr<PeerConnectionBackend> m_backend;

    RTCConfiguration m_configuration;
    RTCController* m_controller { nullptr };
    Vector<RefPtr<RTCCertificate>> m_certificates;
    bool m_shouldDelayTasks { false };
    Deque<std::pair<Ref<DeferredPromise>, Function<void(Ref<DeferredPromise>&&)>>> m_operations;
    bool m_hasPendingOperation { false };
    std::optional<uint32_t> m_negotiationNeededEventId;
    Vector<Ref<RTCDtlsTransport>> m_dtlsTransports;
    Vector<Ref<RTCIceTransport>> m_iceTransports;
    RefPtr<RTCSctpTransport> m_sctpTransport;

    RefPtr<RTCSessionDescription> m_currentLocalDescription;
    RefPtr<RTCSessionDescription> m_pendingLocalDescription;
    RefPtr<RTCSessionDescription> m_currentRemoteDescription;
    RefPtr<RTCSessionDescription> m_pendingRemoteDescription;

    String m_lastCreatedOffer;
    String m_lastCreatedAnswer;
};

} // namespace WebCore

#endif // ENABLE(WEB_RTC)
