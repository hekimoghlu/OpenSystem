/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 1, 2023.
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
#include "api/sctp_transport_interface.h"

#include <optional>
#include <utility>

#include "api/dtls_transport_interface.h"
#include "api/scoped_refptr.h"

namespace webrtc {

SctpTransportInformation::SctpTransportInformation(SctpTransportState state)
    : state_(state) {}

SctpTransportInformation::SctpTransportInformation(
    SctpTransportState state,
    rtc::scoped_refptr<DtlsTransportInterface> dtls_transport,
    std::optional<double> max_message_size,
    std::optional<int> max_channels)
    : state_(state),
      dtls_transport_(std::move(dtls_transport)),
      max_message_size_(max_message_size),
      max_channels_(max_channels) {}

SctpTransportInformation::~SctpTransportInformation() {}

}  // namespace webrtc
