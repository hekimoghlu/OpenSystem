/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 24, 2022.
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
#include "RTCRtpReceiver.h"

#if ENABLE(WEB_RTC)

#include "JSDOMPromiseDeferred.h"
#include "Logging.h"
#include "PeerConnectionBackend.h"
#include "RTCRtpCapabilities.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

#if !RELEASE_LOG_DISABLED
#define LOGIDENTIFIER_RECEIVER Logger::LogSiteIdentifier(logClassName(), __func__, m_connection->logIdentifier())
#else
#define LOGIDENTIFIER_RECEIVER
#endif

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(RTCRtpReceiver);

RTCRtpReceiver::RTCRtpReceiver(PeerConnectionBackend& connection, Ref<MediaStreamTrack>&& track, std::unique_ptr<RTCRtpReceiverBackend>&& backend)
    : m_track(WTFMove(track))
    , m_backend(WTFMove(backend))
    , m_connection(connection)
#if !RELEASE_LOG_DISABLED
    , m_logger(connection.logger())
    , m_logIdentifier(connection.logIdentifier())
#endif
{
}

RTCRtpReceiver::~RTCRtpReceiver()
{
    if (m_transform)
        m_transform->detachFromReceiver(*this);
}

void RTCRtpReceiver::stop()
{
    if (!m_backend)
        return;

    if (m_transform)
        m_transform->detachFromReceiver(*this);

    m_backend = nullptr;
    m_track->stopTrack(MediaStreamTrack::StopMode::PostEvent);
}

void RTCRtpReceiver::getStats(Ref<DeferredPromise>&& promise)
{
    if (!m_connection) {
        promise->reject(ExceptionCode::InvalidStateError);
        return;
    }
    m_connection->getStats(*this, WTFMove(promise));
}

std::optional<RTCRtpCapabilities> RTCRtpReceiver::getCapabilities(ScriptExecutionContext& context, const String& kind)
{
    return PeerConnectionBackend::receiverCapabilities(context, kind);
}

ExceptionOr<void> RTCRtpReceiver::setTransform(std::unique_ptr<RTCRtpTransform>&& transform)
{
    ALWAYS_LOG_IF(m_connection, LOGIDENTIFIER_RECEIVER);

    if (transform && m_transform && *transform == *m_transform)
        return { };
    if (!transform) {
        if (m_transform) {
            m_transform->detachFromReceiver(*this);
            m_transform = { };
        }
        return { };
    }

    if (transform->isAttached())
        return Exception { ExceptionCode::InvalidStateError, "transform is already in use"_s };

    transform->attachToReceiver(*this, m_transform.get());
    m_transform = WTFMove(transform);

    return { };
}

std::optional<RTCRtpTransform::Internal> RTCRtpReceiver::transform()
{
    if (!m_transform)
        return { };
    return m_transform->internalTransform();
}

#if !RELEASE_LOG_DISABLED
WTFLogChannel& RTCRtpReceiver::logChannel() const
{
    return LogWebRTC;
}
#endif

} // namespace WebCore

#endif // ENABLE(WEB_RTC)
