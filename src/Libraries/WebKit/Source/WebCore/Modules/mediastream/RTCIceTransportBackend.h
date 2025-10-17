/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 28, 2024.
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

#include "RTCIceGatheringState.h"
#include "RTCIceTransportState.h"
#include <wtf/WeakPtr.h>

namespace WebCore {
class RTCIceTransportBackendClient;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::RTCIceTransportBackendClient> : std::true_type { };
}

namespace WebCore {

class RTCIceCandidate;

class RTCIceTransportBackendClient : public CanMakeWeakPtr<RTCIceTransportBackendClient> {
public:
    virtual ~RTCIceTransportBackendClient() = default;
    virtual void onStateChanged(RTCIceTransportState) = 0;
    virtual void onGatheringStateChanged(RTCIceGatheringState) = 0;
    virtual void onSelectedCandidatePairChanged(RefPtr<RTCIceCandidate>&&, RefPtr<RTCIceCandidate>&&) = 0;
};

class RTCIceTransportBackend {
public:
    virtual ~RTCIceTransportBackend() = default;
    virtual const void* backend() const = 0;

    virtual void registerClient(RTCIceTransportBackendClient&) = 0;
    virtual void unregisterClient() = 0;
};

inline bool operator==(const RTCIceTransportBackend& a, const RTCIceTransportBackend& b)
{
    return a.backend() == b.backend();
}

} // namespace WebCore

#endif // ENABLE(WEB_RTC)
