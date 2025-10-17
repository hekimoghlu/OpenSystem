/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 19, 2023.
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

#include "RTCSctpTransportState.h"
#include <wtf/WeakPtr.h>

namespace WebCore {
class RTCSctpTransportBackendClient;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::RTCSctpTransportBackendClient> : std::true_type { };
}

namespace WebCore {

class RTCDtlsTransportBackend;

class RTCSctpTransportBackendClient : public CanMakeWeakPtr<RTCSctpTransportBackendClient> {
public:
    virtual ~RTCSctpTransportBackendClient() = default;
    virtual void onStateChanged(RTCSctpTransportState, std::optional<double>, std::optional<unsigned short>) = 0;
};

class RTCSctpTransportBackend {
public:
    virtual ~RTCSctpTransportBackend() = default;

    virtual const void* backend() const = 0;
    virtual UniqueRef<RTCDtlsTransportBackend> dtlsTransportBackend() = 0;

    virtual void registerClient(RTCSctpTransportBackendClient&) = 0;
    virtual void unregisterClient() = 0;
};

inline bool operator==(const RTCSctpTransportBackend& a, const RTCSctpTransportBackend& b)
{
    return a.backend() == b.backend();
}

} // namespace WebCore

#endif // ENABLE(WEB_RTC)
