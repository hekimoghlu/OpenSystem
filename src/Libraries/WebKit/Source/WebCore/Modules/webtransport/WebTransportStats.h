/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 8, 2023.
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

#include "WebTransportDatagramStats.h"

namespace WebCore {

struct WebTransportStats {
    double timestamp { 0 };
    uint64_t bytesSent { 0 };
    uint64_t packetsSent { 0 };
    uint64_t packetsLost { 0 };
    unsigned numOutgoingStreamsCreated { 0 };
    unsigned numIncomingStreamsCreated { 0 };
    uint64_t bytesReceived { 0 };
    uint64_t packetsReceived { 0 };
    double smoothedRtt { 0 };
    double rttVariation { 0 };
    double minRtt { 0 };
    WebTransportDatagramStats datagrams;
    std::optional<uint64_t> estimatedSendRate;
};

}
