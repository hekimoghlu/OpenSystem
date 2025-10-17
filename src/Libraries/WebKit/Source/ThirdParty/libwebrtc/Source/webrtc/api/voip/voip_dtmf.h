/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 29, 2022.
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
#ifndef API_VOIP_VOIP_DTMF_H_
#define API_VOIP_VOIP_DTMF_H_

#include <cstdint>

#include "api/voip/voip_base.h"

namespace webrtc {

// DTMF events and their event codes as defined in
// https://tools.ietf.org/html/rfc4733#section-7
enum class DtmfEvent : uint8_t {
  kDigitZero = 0,
  kDigitOne,
  kDigitTwo,
  kDigitThree,
  kDigitFour,
  kDigitFive,
  kDigitSix,
  kDigitSeven,
  kDigitEight,
  kDigitNine,
  kAsterisk,
  kHash,
  kLetterA,
  kLetterB,
  kLetterC,
  kLetterD
};

// VoipDtmf interface provides DTMF related interfaces such
// as sending DTMF events to the remote endpoint.
class VoipDtmf {
 public:
  // Register the payload type and sample rate for DTMF (RFC 4733) payload.
  // Must be called exactly once prior to calling SendDtmfEvent after payload
  // type has been negotiated with remote.
  // Returns following VoipResult;
  //  kOk - telephone event type is registered as provided.
  //  kInvalidArgument - `channel_id` is invalid.
  virtual VoipResult RegisterTelephoneEventType(ChannelId channel_id,
                                                int rtp_payload_type,
                                                int sample_rate_hz) = 0;

  // Send DTMF named event as specified by
  // https://tools.ietf.org/html/rfc4733#section-3.2
  // `duration_ms` specifies the duration of DTMF packets that will be emitted
  // in place of real RTP packets instead.
  // Must be called after RegisterTelephoneEventType and VoipBase::StartSend
  // have been called.
  // Returns following VoipResult;
  //  kOk - requested DTMF event is successfully scheduled.
  //  kInvalidArgument - `channel_id` is invalid.
  //  kFailedPrecondition - Missing prerequisite on RegisterTelephoneEventType
  //   or sending state.
  virtual VoipResult SendDtmfEvent(ChannelId channel_id,
                                   DtmfEvent dtmf_event,
                                   int duration_ms) = 0;

 protected:
  virtual ~VoipDtmf() = default;
};

}  // namespace webrtc

#endif  // API_VOIP_VOIP_DTMF_H_
