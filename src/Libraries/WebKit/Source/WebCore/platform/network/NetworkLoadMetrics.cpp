/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 29, 2025.
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
#include "NetworkLoadMetrics.h"

#include <wtf/NeverDestroyed.h>

namespace WebCore {

NetworkLoadMetrics::NetworkLoadMetrics() = default;

NetworkLoadMetrics::NetworkLoadMetrics(MonotonicTime&& redirectStart, MonotonicTime&& fetchStart, MonotonicTime&& domainLookupStart, MonotonicTime&& domainLookupEnd, MonotonicTime&& connectStart, MonotonicTime&& secureConnectionStart, MonotonicTime&& connectEnd, MonotonicTime&& requestStart, MonotonicTime&& responseStart, MonotonicTime&& responseEnd, MonotonicTime&& workerStart, String&& protocol, uint16_t redirectCount, bool complete, bool cellular, bool expensive, bool constrained, bool multipath, bool isReusedConnection, bool failsTAOCheck, bool hasCrossOriginRedirect, PrivacyStance privacyStance, uint64_t responseBodyBytesReceived, uint64_t responseBodyDecodedSize, RefPtr<AdditionalNetworkLoadMetricsForWebInspector>&& additionalNetworkLoadMetricsForWebInspector)
    : redirectStart(WTFMove(redirectStart))
    , fetchStart(WTFMove(fetchStart))
    , domainLookupStart(WTFMove(domainLookupStart))
    , domainLookupEnd(WTFMove(domainLookupEnd))
    , connectStart(WTFMove(connectStart))
    , secureConnectionStart(WTFMove(secureConnectionStart))
    , connectEnd(WTFMove(connectEnd))
    , requestStart(WTFMove(requestStart))
    , responseStart(WTFMove(responseStart))
    , responseEnd(responseEnd)
    , workerStart(workerStart)
    , protocol(protocol)
    , redirectCount(redirectCount)
    , complete(complete)
    , cellular(cellular)
    , expensive(expensive)
    , constrained(constrained)
    , multipath(multipath)
    , isReusedConnection(isReusedConnection)
    , failsTAOCheck(failsTAOCheck)
    , hasCrossOriginRedirect(hasCrossOriginRedirect)
    , privacyStance(privacyStance)
    , responseBodyBytesReceived(responseBodyBytesReceived)
    , responseBodyDecodedSize(responseBodyDecodedSize)
    , additionalNetworkLoadMetricsForWebInspector(WTFMove(additionalNetworkLoadMetricsForWebInspector))
{
}

void NetworkLoadMetrics::updateFromFinalMetrics(const NetworkLoadMetrics& other)
{
    MonotonicTime originalRedirectStart = redirectStart;
    MonotonicTime originalFetchStart = fetchStart;
    MonotonicTime originalDomainLookupStart = domainLookupStart;
    MonotonicTime originalDomainLookupEnd = domainLookupEnd;
    MonotonicTime originalConnectStart = connectStart;
    MonotonicTime originalSecureConnectionStart = secureConnectionStart;
    MonotonicTime originalConnectEnd = connectEnd;
    MonotonicTime originalRequestStart = requestStart;
    MonotonicTime originalResponseStart = responseStart;
    MonotonicTime originalResponseEnd = responseEnd;
    MonotonicTime originalWorkerStart = workerStart;

    *this = other;

    if (!redirectStart)
        redirectStart = originalRedirectStart;
    if (!fetchStart)
        fetchStart = originalFetchStart;
    if (!domainLookupStart)
        domainLookupStart = originalDomainLookupStart;
    if (!domainLookupEnd)
        domainLookupEnd = originalDomainLookupEnd;
    if (!connectStart)
        connectStart = originalConnectStart;
    if (!secureConnectionStart)
        secureConnectionStart = originalSecureConnectionStart;
    if (!connectEnd)
        connectEnd = originalConnectEnd;
    if (!requestStart)
        requestStart = originalRequestStart;
    if (!responseStart)
        responseStart = originalResponseStart;
    if (!responseEnd)
        responseEnd = originalResponseEnd;
    if (!workerStart)
        workerStart = originalWorkerStart;

    if (!responseEnd)
        responseEnd = MonotonicTime::now();
    complete = true;
}

const NetworkLoadMetrics& NetworkLoadMetrics::emptyMetrics()
{
    static NeverDestroyed<NetworkLoadMetrics> metrics;
    return metrics.get();
}

Ref<AdditionalNetworkLoadMetricsForWebInspector> AdditionalNetworkLoadMetricsForWebInspector::isolatedCopy()
{
    auto copy = AdditionalNetworkLoadMetricsForWebInspector::create();
    copy->priority = priority;
    copy->remoteAddress = remoteAddress.isolatedCopy();
    copy->connectionIdentifier = connectionIdentifier.isolatedCopy();
    copy->tlsProtocol = tlsProtocol.isolatedCopy();
    copy->tlsCipher = tlsCipher.isolatedCopy();
    copy->requestHeaders = requestHeaders.isolatedCopy();
    copy->requestHeaderBytesSent = requestHeaderBytesSent;
    copy->responseHeaderBytesReceived = responseHeaderBytesReceived;
    copy->requestBodyBytesSent = requestBodyBytesSent;
    copy->isProxyConnection = isProxyConnection;
    return copy;
}

NetworkLoadMetrics NetworkLoadMetrics::isolatedCopy() const
{
    NetworkLoadMetrics copy;

    copy.redirectStart = redirectStart.isolatedCopy();
    copy.fetchStart = fetchStart.isolatedCopy();
    copy.domainLookupStart = domainLookupStart.isolatedCopy();
    copy.domainLookupEnd = domainLookupEnd.isolatedCopy();
    copy.connectStart = connectStart.isolatedCopy();
    copy.secureConnectionStart = secureConnectionStart.isolatedCopy();
    copy.connectEnd = connectEnd.isolatedCopy();
    copy.requestStart = requestStart.isolatedCopy();
    copy.responseStart = responseStart.isolatedCopy();
    copy.responseEnd = responseEnd.isolatedCopy();
    copy.workerStart = workerStart.isolatedCopy();

    copy.protocol = protocol.isolatedCopy();

    copy.redirectCount = redirectCount;

    copy.complete = complete;
    copy.cellular = cellular;
    copy.expensive = expensive;
    copy.constrained = constrained;
    copy.multipath = multipath;
    copy.isReusedConnection = isReusedConnection;
    copy.failsTAOCheck = failsTAOCheck;
    copy.hasCrossOriginRedirect = hasCrossOriginRedirect;

    copy.privacyStance = privacyStance;

    copy.responseBodyBytesReceived = responseBodyBytesReceived;
    copy.responseBodyDecodedSize = responseBodyDecodedSize;

    if (additionalNetworkLoadMetricsForWebInspector)
        copy.additionalNetworkLoadMetricsForWebInspector = additionalNetworkLoadMetricsForWebInspector->isolatedCopy();

    return copy;
}

Ref<AdditionalNetworkLoadMetricsForWebInspector> AdditionalNetworkLoadMetricsForWebInspector::create(NetworkLoadPriority&& priority, String&& remoteAddress, String&& connectionIdentifier, String&& tlsProtocol, String&& tlsCipher, HTTPHeaderMap&& requestHeaders, uint64_t requestHeaderBytesSent, uint64_t responseHeaderBytesReceived, uint64_t requestBodyBytesSent, bool isProxyConnection)
{
    return adoptRef(*new AdditionalNetworkLoadMetricsForWebInspector(WTFMove(priority), WTFMove(remoteAddress), WTFMove(connectionIdentifier), WTFMove(tlsProtocol), WTFMove(tlsCipher), WTFMove(requestHeaders), requestHeaderBytesSent, responseHeaderBytesReceived, requestBodyBytesSent, isProxyConnection));
}

AdditionalNetworkLoadMetricsForWebInspector::AdditionalNetworkLoadMetricsForWebInspector(NetworkLoadPriority&& priority, String&& remoteAddress, String&& connectionIdentifier, String&& tlsProtocol, String&& tlsCipher, HTTPHeaderMap&& requestHeaders, uint64_t requestHeaderBytesSent, uint64_t responseHeaderBytesReceived, uint64_t requestBodyBytesSent, bool isProxyConnection)
    : priority(WTFMove(priority))
    , remoteAddress(WTFMove(remoteAddress))
    , connectionIdentifier(WTFMove(connectionIdentifier))
    , tlsProtocol(WTFMove(tlsProtocol))
    , tlsCipher(WTFMove(tlsCipher))
    , requestHeaders(WTFMove(requestHeaders))
    , requestHeaderBytesSent(requestHeaderBytesSent)
    , responseHeaderBytesReceived(responseHeaderBytesReceived)
    , requestBodyBytesSent(requestBodyBytesSent)
    , isProxyConnection(isProxyConnection)
{

}

} // namespace WebCore
