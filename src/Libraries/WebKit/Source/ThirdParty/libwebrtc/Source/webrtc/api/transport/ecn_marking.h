/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 21, 2024.
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
#ifndef API_TRANSPORT_ECN_MARKING_H_
#define API_TRANSPORT_ECN_MARKING_H_

namespace webrtc {

// TODO: bugs.webrtc.org/42225697 - L4S support is slowly being developed.
// Help is appreciated.

// L4S Explicit Congestion Notification (ECN) .
// https://www.rfc-editor.org/rfc/rfc9331.html ECT stands for ECN-Capable
// Transport and CE stands for Congestion Experienced.

// RFC-3168, Section 5
// +-----+-----+
// | ECN FIELD |
// +-----+-----+
//   ECT   CE         [Obsolete] RFC 2481 names for the ECN bits.
//    0     0         Not-ECT
//    0     1         ECT(1)
//    1     0         ECT(0)
//    1     1         CE

enum class EcnMarking {
  kNotEct = 0,  // Not ECN-Capable Transport
  kEct1 = 1,    // ECN-Capable Transport
  kEct0 = 2,    // Not used by L4s (or webrtc.)
  kCe = 3,      // Congestion experienced
};

}  // namespace webrtc

#endif  // API_TRANSPORT_ECN_MARKING_H_
