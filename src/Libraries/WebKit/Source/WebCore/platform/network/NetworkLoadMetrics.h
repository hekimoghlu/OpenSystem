/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 5, 2025.
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

#include "HTTPHeaderMap.h"
#include <wtf/Box.h>
#include <wtf/MonotonicTime.h>
#include <wtf/text/WTFString.h>

#if PLATFORM(COCOA)
OBJC_CLASS NSURLConnection;
OBJC_CLASS NSURLSessionTaskMetrics;
#endif

namespace WebCore {

class ResourceHandle;

enum class NetworkLoadPriority : uint8_t {
    Low,
    Medium,
    High,
    Unknown,
};

enum class PrivacyStance : uint8_t {
    Unknown,
    NotEligible,
    Proxied,
    Failed,
    Direct,
    FailedUnreachable,
};

constexpr MonotonicTime reusedTLSConnectionSentinel { MonotonicTime::fromRawSeconds(-1) };

struct AdditionalNetworkLoadMetricsForWebInspector;

class NetworkLoadMetrics {
    WTF_MAKE_FAST_ALLOCATED(NetworkLoadMetrics);
public:
    WEBCORE_EXPORT NetworkLoadMetrics();
    WEBCORE_EXPORT NetworkLoadMetrics(MonotonicTime&& redirectStart, MonotonicTime&& fetchStart, MonotonicTime&& domainLookupStart, MonotonicTime&& domainLookupEnd, MonotonicTime&& connectStart, MonotonicTime&& secureConnectionStart, MonotonicTime&& connectEnd, MonotonicTime&& requestStart, MonotonicTime&& responseStart, MonotonicTime&& responseEnd, MonotonicTime&& workerStart, String&& protocol, uint16_t redirectCount, bool complete, bool cellular, bool expensive, bool constrained, bool multipath, bool isReusedConnection, bool failsTAOCheck, bool hasCrossOriginRedirect, PrivacyStance, uint64_t responseBodyBytesReceived, uint64_t responseBodyDecodedSize, RefPtr<AdditionalNetworkLoadMetricsForWebInspector>&&);

    WEBCORE_EXPORT static const NetworkLoadMetrics& emptyMetrics();

    WEBCORE_EXPORT NetworkLoadMetrics isolatedCopy() const;

    bool isComplete() const { return complete; }
    bool isCellular() const { return cellular; }
    bool isExpensive() const { return expensive; }
    bool isConstrained() const { return constrained; }
    bool isMultipath() const { return multipath; }
    bool reusedConnection() const { return isReusedConnection; }
    bool doesFailTAOCheck() const { return failsTAOCheck; }
    bool crossOriginRedirect() const { return hasCrossOriginRedirect; }
    void markComplete() { complete = true; }

    void updateFromFinalMetrics(const NetworkLoadMetrics&);

    // https://www.w3.org/TR/resource-timing-2/#attribute-descriptions
    MonotonicTime redirectStart;
    MonotonicTime fetchStart;
    MonotonicTime domainLookupStart;
    MonotonicTime domainLookupEnd;
    MonotonicTime connectStart;
    MonotonicTime secureConnectionStart;
    MonotonicTime connectEnd;
    MonotonicTime requestStart;
    MonotonicTime responseStart;
    MonotonicTime responseEnd;
    MonotonicTime workerStart;

    // ALPN Protocol ID: https://w3c.github.io/resource-timing/#bib-RFC7301
    String protocol;

    uint16_t redirectCount { 0 };

    bool complete : 1 { false };
    bool cellular : 1 { false };
    bool expensive : 1 { false };
    bool constrained : 1 { false };
    bool multipath : 1 { false };
    bool isReusedConnection : 1 { false };
    bool failsTAOCheck : 1 { false };
    bool hasCrossOriginRedirect : 1 { false };

    PrivacyStance privacyStance { PrivacyStance::Unknown };

    uint64_t responseBodyBytesReceived { std::numeric_limits<uint64_t>::max() };
    uint64_t responseBodyDecodedSize { std::numeric_limits<uint64_t>::max() };

    RefPtr<AdditionalNetworkLoadMetricsForWebInspector> additionalNetworkLoadMetricsForWebInspector;
};

struct AdditionalNetworkLoadMetricsForWebInspector : public RefCounted<AdditionalNetworkLoadMetricsForWebInspector> {

    static Ref<AdditionalNetworkLoadMetricsForWebInspector> create() { return adoptRef(*new AdditionalNetworkLoadMetricsForWebInspector()); }
    WEBCORE_EXPORT static Ref<AdditionalNetworkLoadMetricsForWebInspector> create(NetworkLoadPriority&&, String&& remoteAddress, String&& connectionIdentifier, String&& tlsProtocol, String&& tlsCipher, HTTPHeaderMap&& requestHeaders, uint64_t requestHeaderBytesSent, uint64_t responseHeaderBytesReceived, uint64_t requestBodyBytesSent, bool isProxyConnection);
    Ref<AdditionalNetworkLoadMetricsForWebInspector> isolatedCopy() const;
    Ref<AdditionalNetworkLoadMetricsForWebInspector> isolatedCopy();

    NetworkLoadPriority priority { NetworkLoadPriority::Unknown };

    String remoteAddress;
    String connectionIdentifier;

    String tlsProtocol;
    String tlsCipher;

    HTTPHeaderMap requestHeaders;

    uint64_t requestHeaderBytesSent { std::numeric_limits<uint64_t>::max() };
    uint64_t responseHeaderBytesReceived { std::numeric_limits<uint64_t>::max() };
    uint64_t requestBodyBytesSent { std::numeric_limits<uint64_t>::max() };

    bool isProxyConnection { false };
private:
    AdditionalNetworkLoadMetricsForWebInspector() { }
    AdditionalNetworkLoadMetricsForWebInspector(NetworkLoadPriority&&, String&& remoteAddress, String&& connectionIdentifier, String&& tlsProtocol, String&& tlsCipher, HTTPHeaderMap&& requestHeaders, uint64_t requestHeaderBytesSent, uint64_t responseHeaderBytesReceived, uint64_t requestBodyBytesSent, bool isProxyConnection);
};

#if PLATFORM(COCOA)
Box<NetworkLoadMetrics> copyTimingData(NSURLConnection *, const ResourceHandle&);
WEBCORE_EXPORT Box<NetworkLoadMetrics> copyTimingData(NSURLSessionTaskMetrics *incompleteMetrics, const NetworkLoadMetrics&);
#endif

} // namespace WebCore
