/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 4, 2023.
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
#ifndef PC_SIMULCAST_SDP_SERIALIZER_H_
#define PC_SIMULCAST_SDP_SERIALIZER_H_

#include <string>

#include "absl/strings/string_view.h"
#include "api/rtc_error.h"
#include "media/base/rid_description.h"
#include "pc/session_description.h"
#include "pc/simulcast_description.h"

namespace webrtc {

// This class serializes simulcast components of the SDP.
// Example:
//     SimulcastDescription can be serialized and deserialized by this class.
//     The serializer will know how to translate the data to spec-compliant
//     format without knowing about the SDP attribute details (a=simulcast:)
// Usage:
//     Consider the SDP attribute for simulcast a=simulcast:<configuration>.
//     The SDP serializtion code (webrtc_sdp.h) should use `SdpSerializer` to
//     serialize and deserialize the <configuration> section.
// This class will allow testing the serialization of components without
// having to serialize the entire SDP while hiding implementation details
// from callers of sdp serialization (webrtc_sdp.h).
class SimulcastSdpSerializer {
 public:
  // Serialization for the Simulcast description according to
  // https://tools.ietf.org/html/draft-ietf-mmusic-sdp-simulcast-13#section-5.1
  std::string SerializeSimulcastDescription(
      const cricket::SimulcastDescription& simulcast) const;

  // Deserialization for the SimulcastDescription according to
  // https://tools.ietf.org/html/draft-ietf-mmusic-sdp-simulcast-13#section-5.1
  RTCErrorOr<cricket::SimulcastDescription> DeserializeSimulcastDescription(
      absl::string_view string) const;

  // Serialization for the RID description according to
  // https://tools.ietf.org/html/draft-ietf-mmusic-rid-15#section-10
  std::string SerializeRidDescription(
      const cricket::RidDescription& rid_description) const;

  // Deserialization for the RidDescription according to
  // https://tools.ietf.org/html/draft-ietf-mmusic-rid-15#section-10
  RTCErrorOr<cricket::RidDescription> DeserializeRidDescription(
      absl::string_view string) const;
};

}  // namespace webrtc

#endif  // PC_SIMULCAST_SDP_SERIALIZER_H_
