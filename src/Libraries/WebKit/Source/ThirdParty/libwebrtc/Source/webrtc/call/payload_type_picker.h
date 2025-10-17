/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 25, 2022.
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
#ifndef CALL_PAYLOAD_TYPE_PICKER_H_
#define CALL_PAYLOAD_TYPE_PICKER_H_

#include <map>
#include <set>
#include <utility>
#include <vector>

#include "api/rtc_error.h"
#include "call/payload_type.h"
#include "media/base/codec.h"
#include "rtc_base/checks.h"

namespace webrtc {

class PayloadTypeRecorder;

class PayloadTypePicker {
 public:
  PayloadTypePicker();
  PayloadTypePicker(const PayloadTypePicker&) = delete;
  PayloadTypePicker& operator=(const PayloadTypePicker&) = delete;
  PayloadTypePicker(PayloadTypePicker&&) = delete;
  PayloadTypePicker& operator=(PayloadTypePicker&&) = delete;
  // Suggest a payload type for the codec.
  // If the excluder maps it to something different, don't suggest it.
  RTCErrorOr<PayloadType> SuggestMapping(cricket::Codec codec,
                                         const PayloadTypeRecorder* excluder);
  RTCError AddMapping(PayloadType payload_type, cricket::Codec codec);

 private:
  class MapEntry {
   public:
    MapEntry(PayloadType payload_type, cricket::Codec codec)
        : payload_type_(payload_type), codec_(codec) {}
    PayloadType payload_type() const { return payload_type_; }
    cricket::Codec codec() const { return codec_; }

   private:
    PayloadType payload_type_;
    cricket::Codec codec_;
  };
  std::vector<MapEntry> entries_;
  std::set<PayloadType> seen_payload_types_;
};

class PayloadTypeRecorder {
 public:
  explicit PayloadTypeRecorder(PayloadTypePicker& suggester)
      : suggester_(suggester) {}
  ~PayloadTypeRecorder() {
    // Ensure consistent use of paired Disallow/ReallowRedefintion calls.
    RTC_DCHECK(disallow_redefinition_level_ == 0);
  }

  RTCError AddMapping(PayloadType payload_type, cricket::Codec codec);
  std::vector<std::pair<PayloadType, cricket::Codec>> GetMappings() const;
  RTCErrorOr<PayloadType> LookupPayloadType(cricket::Codec codec) const;
  RTCErrorOr<cricket::Codec> LookupCodec(PayloadType payload_type) const;
  // Redefinition guard.
  // In some scenarios, redefinition must be allowed between one offer/answer
  // set and the next offer/answer set, but within the processing of one
  // SDP, it should never be allowed.
  // Implemented as a stack push/pop for convenience; if Disallow has
  // been called more times than Reallow, redefinition is prohibited.
  void DisallowRedefinition();
  void ReallowRedefinition();
  // Transaction support.
  // Commit() commits previous changes.
  void Commit();
  // Rollback() rolls back to the previous checkpoint.
  void Rollback();

 private:
  PayloadTypePicker& suggester_;
  std::map<PayloadType, cricket::Codec> payload_type_to_codec_;
  std::map<PayloadType, cricket::Codec> checkpoint_payload_type_to_codec_;
  int disallow_redefinition_level_ = 0;
  std::set<PayloadType> accepted_definitions_;
};

}  // namespace webrtc

#endif  //  CALL_PAYLOAD_TYPE_PICKER_H_
