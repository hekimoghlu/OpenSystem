/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 28, 2025.
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

#include "RTCIceCandidateType.h"
#include "RTCIceComponent.h"
#include "RTCIceProtocol.h"
#include "RTCIceTcpCandidateType.h"
#include <optional>
#include <wtf/text/WTFString.h>

namespace WebCore {

struct RTCIceCandidateFields {
    String foundation;
    std::optional<RTCIceComponent> component;
    std::optional<unsigned> priority;
    String address;
    std::optional<RTCIceProtocol> protocol;
    std::optional<unsigned short> port;
    std::optional<RTCIceCandidateType> type;
    std::optional<RTCIceTcpCandidateType> tcpType;
    String relatedAddress;
    std::optional<unsigned short> relatedPort;
    String usernameFragment;

    RTCIceCandidateFields isolatedCopy() && { return { WTFMove(foundation).isolatedCopy(), component, priority, WTFMove(address).isolatedCopy(), protocol, port, type, tcpType, WTFMove(relatedAddress).isolatedCopy(), relatedPort, WTFMove(usernameFragment).isolatedCopy() }; }
};

std::optional<RTCIceCandidateFields> parseIceCandidateSDP(const String&);

} // namespace WebCore
