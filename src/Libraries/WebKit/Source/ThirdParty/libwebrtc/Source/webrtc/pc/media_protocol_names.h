/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 28, 2022.
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
#ifndef PC_MEDIA_PROTOCOL_NAMES_H_
#define PC_MEDIA_PROTOCOL_NAMES_H_

#include "absl/strings/string_view.h"

namespace cricket {

// Names or name prefixes of protocols as defined by SDP specifications,
// and generated in SDP produced by WebRTC.
extern const char kMediaProtocolSctp[];
extern const char kMediaProtocolUdpDtlsSctp[];
extern const char kMediaProtocolDtlsSavpf[];
extern const char kMediaProtocolSavpf[];
extern const char kMediaProtocolAvpf[];

// Exported for testing only
extern const char kMediaProtocolTcpDtlsSctp[];
extern const char kMediaProtocolDtlsSctp[];

// Returns true if the given media section protocol indicates use of RTP.
bool IsRtpProtocol(absl::string_view protocol);
// Returns true if the given media section protocol indicates use of SCTP.
bool IsSctpProtocol(absl::string_view protocol);

// Returns true if the given media protocol is unencrypted SCTP
bool IsPlainSctp(absl::string_view protocol);
// Returns true if the given media protocol is encrypted SCTP
bool IsDtlsSctp(absl::string_view protocol);

// Returns true if the given media protocol is unencrypted RTP
bool IsPlainRtp(absl::string_view protocol);
// Returns true if the given media protocol is encrypted RTP
bool IsDtlsRtp(absl::string_view protocol);

}  // namespace cricket

#endif  // PC_MEDIA_PROTOCOL_NAMES_H_
