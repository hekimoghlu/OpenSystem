/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 22, 2022.
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
#ifndef MEDIA_BASE_RID_DESCRIPTION_H_
#define MEDIA_BASE_RID_DESCRIPTION_H_

#include <map>
#include <string>
#include <vector>

namespace cricket {

enum class RidDirection { kSend, kReceive };

// Description of a Restriction Id (RID) according to:
// https://tools.ietf.org/html/draft-ietf-mmusic-rid-15
// A Restriction Identifier serves two purposes:
//   1. Uniquely identifies an RTP stream inside an RTP session.
//      When combined with MIDs (https://tools.ietf.org/html/rfc5888),
//      RIDs uniquely identify an RTP stream within an RTP session.
//      The MID will identify the media section and the RID will identify
//      the stream within the section.
//      RID identifiers must be unique within the media section.
//   2. Allows indicating further restrictions to the stream.
//      These restrictions are added according to the direction specified.
//      The direction field identifies the direction of the RTP stream packets
//      to which the restrictions apply. The direction is independent of the
//      transceiver direction and can be one of {send, recv}.
//      The following are some examples of these restrictions:
//        a. max-width, max-height, max-fps, max-br, ...
//        b. further restricting the codec set (from what m= section specified)
//
// Note: Indicating dependencies between streams (using depend) will not be
// supported, since the WG is adopting a different approach to achieve this.
// As of 2018-12-04, the new SVC (Scalable Video Coder) approach is still not
// mature enough to be implemented as part of this work.
// See: https://w3c.github.io/webrtc-svc/ for more details.
struct RidDescription final {
  RidDescription();
  RidDescription(const std::string& rid, RidDirection direction);
  RidDescription(const RidDescription& other);
  ~RidDescription();
  RidDescription& operator=(const RidDescription& other);

  // This is currently required for unit tests of StreamParams which contains
  // RidDescription objects and checks for equality using operator==.
  bool operator==(const RidDescription& other) const;
  bool operator!=(const RidDescription& other) const {
    return !(*this == other);
  }

  // The RID identifier that uniquely identifies the stream within the session.
  std::string rid;

  // Specifies the direction for which the specified restrictions hold.
  // This direction is either send or receive and is independent of the
  // direction of the transceiver.
  // https://tools.ietf.org/html/draft-ietf-mmusic-rid-15#section-4 :
  // The "direction" field identifies the direction of the RTP Stream
  // packets to which the indicated restrictions are applied.  It may be
  // either "send" or "recv".  Note that these restriction directions are
  // expressed independently of any "inactive", "sendonly", "recvonly", or
  // "sendrecv" attributes associated with the media section.  It is, for
  // example, valid to indicate "recv" restrictions on a "sendonly"
  // stream; those restrictions would apply if, at a future point in time,
  // the stream were changed to "sendrecv" or "recvonly".
  RidDirection direction;

  // The list of codec payload types for this stream.
  // It should be a subset of the payloads supported for the media section.
  std::vector<int> payload_types;

  // Contains key-value pairs for restrictions.
  // The keys are not validated against a known set.
  // The meaning to infer for the values depends on each key.
  // Examples:
  // 1. An entry for max-width will have a value that is interpreted as an int.
  // 2. An entry for max-bpp (bits per pixel) will have a float value.
  // Interpretation (and validation of value) is left for the implementation.
  // I.E. the media engines should validate values for parameters they support.
  std::map<std::string, std::string> restrictions;
};

}  // namespace cricket

#endif  // MEDIA_BASE_RID_DESCRIPTION_H_
