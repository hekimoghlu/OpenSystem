/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 2, 2022.
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
#ifndef CALL_PAYLOAD_TYPE_H_
#define CALL_PAYLOAD_TYPE_H_

#include <cstdint>
#include <string>

#include "api/rtc_error.h"
#include "media/base/codec.h"
#include "rtc_base/strong_alias.h"

namespace webrtc {

class PayloadType : public StrongAlias<class PayloadTypeTag, uint8_t> {
 public:
  // Non-explicit conversions from and to ints are to be deprecated and
  // removed once calling code is upgraded.
  PayloadType(uint8_t pt) { value_ = pt; }                // NOLINT: explicit
  constexpr operator uint8_t() const& { return value_; }  // NOLINT: Explicit
};

class PayloadTypeSuggester {
 public:
  virtual ~PayloadTypeSuggester() = default;
  // Suggest a payload type for a given codec on a given media section.
  // Media section is indicated by MID.
  // The function will either return a PT already in use on the connection
  // or a newly suggested one.
  virtual RTCErrorOr<PayloadType> SuggestPayloadType(const std::string& mid,
                                                     cricket::Codec codec) = 0;
  // Register a payload type as mapped to a specific codec for this MID
  // at this time.
  virtual RTCError AddLocalMapping(const std::string& mid,
                                   PayloadType payload_type,
                                   const cricket::Codec& codec) = 0;
};

}  // namespace webrtc

#endif  // CALL_PAYLOAD_TYPE_H_
