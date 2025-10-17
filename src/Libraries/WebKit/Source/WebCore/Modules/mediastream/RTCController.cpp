/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 27, 2021.
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
#include "RTCController.h"

#if ENABLE(WEB_RTC)

#include "DocumentInlines.h"
#include "RTCNetworkManager.h"
#include "RTCPeerConnection.h"
#include "WebRTCProvider.h"

#if USE(LIBWEBRTC)
#include "LibWebRTCLogSink.h"
#include "LibWebRTCUtils.h"
#endif

#if USE(GSTREAMER_WEBRTC)
#include "GStreamerWebRTCLogSink.h"
#endif

#endif

namespace WebCore {

RTCController::RTCController()
{
}

#if ENABLE(WEB_RTC)

RTCController::~RTCController()
{
    for (Ref connection : m_peerConnections)
        connection->clearController();
    stopGatheringLogs();
}

void RTCController::reset(bool shouldFilterICECandidates)
{
    m_shouldFilterICECandidates = shouldFilterICECandidates;
    for (Ref connection : m_peerConnections)
        connection->clearController();
    m_peerConnections.clear();
    m_filteringDisabledOrigins.clear();
}

void RTCController::remove(RTCPeerConnection& connection)
{
    m_peerConnections.remove(connection);
}

static inline bool matchDocumentOrigin(Document& document, SecurityOrigin& topOrigin, SecurityOrigin& clientOrigin)
{
    if (topOrigin.isSameOriginAs(document.protectedSecurityOrigin()))
        return true;
    return topOrigin.isSameOriginAs(document.protectedTopOrigin()) && clientOrigin.isSameOriginAs(document.protectedSecurityOrigin());
}

bool RTCController::shouldDisableICECandidateFiltering(Document& document)
{
    if (!m_shouldFilterICECandidates)
        return true;
    return notFound != m_filteringDisabledOrigins.findIf([&] (const auto& origin) {
        return matchDocumentOrigin(document, origin.topOrigin, origin.clientOrigin);
    });
}

void RTCController::add(RTCPeerConnection& connection)
{
    Ref document = downcast<Document>(*connection.scriptExecutionContext());
    if (RefPtr networkManager = document->rtcNetworkManager())
        networkManager->setICECandidateFiltering(!shouldDisableICECandidateFiltering(document));

    m_peerConnections.add(connection);
    if (shouldDisableICECandidateFiltering(downcast<Document>(*connection.scriptExecutionContext())))
        connection.disableICECandidateFiltering();

    if (m_gatheringLogsDocument && connection.scriptExecutionContext() == m_gatheringLogsDocument.get())
        startGatheringStatLogs(connection);
}

void RTCController::disableICECandidateFilteringForAllOrigins()
{
    if (!WebRTCProvider::webRTCAvailable())
        return;

    m_shouldFilterICECandidates = false;
    for (Ref connection : m_peerConnections) {
        if (RefPtr document = downcast<Document>(connection->scriptExecutionContext())) {
            if (RefPtr networkManager = document->rtcNetworkManager())
                networkManager->setICECandidateFiltering(false);
        }
        connection->disableICECandidateFiltering();
    }
}

void RTCController::disableICECandidateFilteringForDocument(Document& document)
{
    if (!WebRTCProvider::webRTCAvailable())
        return;

    if (RefPtr networkManager = document.rtcNetworkManager())
        networkManager->setICECandidateFiltering(false);

    m_filteringDisabledOrigins.append(PeerConnectionOrigin { document.topOrigin(), document.securityOrigin() });
    for (Ref connection : m_peerConnections) {
        if (RefPtr peerConnectionDocument = downcast<Document>(connection->scriptExecutionContext())) {
            if (matchDocumentOrigin(*peerConnectionDocument, document.topOrigin(), document.securityOrigin())) {
                if (RefPtr networkManager = peerConnectionDocument->rtcNetworkManager())
                    networkManager->setICECandidateFiltering(false);
                connection->disableICECandidateFiltering();
            }
        }
    }
}

void RTCController::enableICECandidateFiltering()
{
    if (!WebRTCProvider::webRTCAvailable())
        return;

    m_filteringDisabledOrigins.clear();
    m_shouldFilterICECandidates = true;
    for (Ref connection : m_peerConnections) {
        connection->enableICECandidateFiltering();
        if (RefPtr document = downcast<Document>(connection->scriptExecutionContext())) {
            if (RefPtr networkManager = document->rtcNetworkManager())
                networkManager->setICECandidateFiltering(true);
        }
    }
}

#if USE(LIBWEBRTC)
static String toWebRTCLogLevel(rtc::LoggingSeverity severity)
{
    switch (severity) {
    case rtc::LoggingSeverity::LS_VERBOSE:
        return "verbose"_s;
    case rtc::LoggingSeverity::LS_INFO:
        return "info"_s;
    case rtc::LoggingSeverity::LS_WARNING:
        return "warning"_s;
    case rtc::LoggingSeverity::LS_ERROR:
        return "error"_s;
    case rtc::LoggingSeverity::LS_NONE:
        return "none"_s;
    }
    ASSERT_NOT_REACHED();
    return ""_s;
}
#endif

void RTCController::startGatheringLogs(Document& document, LogCallback&& callback)
{
    m_gatheringLogsDocument = document;
    m_callback = WTFMove(callback);
    for (Ref connection : m_peerConnections) {
        if (connection->scriptExecutionContext() != &document) {
            connection->stopGatheringStatLogs();
            continue;
        }
        startGatheringStatLogs(connection);
    }
#if USE(LIBWEBRTC)
    if (!m_logSink) {
        m_logSink = makeUnique<LibWebRTCLogSink>([weakThis = WeakPtr { *this }] (auto&& logLevel, auto&& logMessage) {
            ensureOnMainThread([weakThis, logMessage = fromStdString(logMessage).isolatedCopy(), logLevel] () mutable {
                if (auto protectedThis = weakThis.get())
                    protectedThis->m_callback("backend-logs"_s, WTFMove(logMessage), toWebRTCLogLevel(logLevel), nullptr);
            });
        });
        m_logSink->start();
    }
#endif

#if USE(GSTREAMER_WEBRTC)
    if (!m_logSink) {
        m_logSink = makeUnique<GStreamerWebRTCLogSink>([weakThis = WeakPtr { *this }](const auto& logLevel, const auto& logMessage) {
            ensureOnMainThread([weakThis, logMessage = logMessage.isolatedCopy(), logLevel = logLevel.isolatedCopy()]() mutable {
                if (auto protectedThis = weakThis.get())
                    protectedThis->m_callback("backend-logs"_s, WTFMove(logMessage), WTFMove(logLevel), nullptr);
            });
        });
        m_logSink->start();
    }
#endif
}

void RTCController::stopGatheringLogs()
{
    if (!m_gatheringLogsDocument)
        return;
    m_gatheringLogsDocument = { };
    m_callback = { };

    for (Ref connection : m_peerConnections)
        connection->stopGatheringStatLogs();

    stopLoggingWebRTCLogs();
}

void RTCController::startGatheringStatLogs(RTCPeerConnection& connection)
{
    connection.startGatheringStatLogs([weakThis = WeakPtr { *this }, connection = WeakPtr { connection }] (auto&& stats) {
        if (weakThis)
            weakThis->m_callback("stats"_s, WTFMove(stats), { }, connection.get());
    });
}

void RTCController::stopLoggingWebRTCLogs()
{
#if USE(LIBWEBRTC) || USE(GSTREAMER_WEBRTC)
    if (!m_logSink)
        return;

    m_logSink->stop();
    m_logSink = nullptr;
#endif
}

#endif // ENABLE(WEB_RTC)

} // namespace WebCore
