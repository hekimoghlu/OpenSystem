/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 7, 2024.
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
#ifndef API_RTC_EVENT_LOG_RTC_EVENT_H_
#define API_RTC_EVENT_LOG_RTC_EVENT_H_

#include <cstdint>

namespace webrtc {

// This class allows us to store unencoded RTC events. Subclasses of this class
// store the actual information. This allows us to keep all unencoded events,
// even when their type and associated information differ, in the same buffer.
// Additionally, it prevents dependency leaking - a module that only logs
// events of type RtcEvent_A doesn't need to know about anything associated
// with events of type RtcEvent_B.
class RtcEvent {
 public:
  // Subclasses of this class have to associate themselves with a unique value
  // of Type. This leaks the information of existing subclasses into the
  // superclass, but the *actual* information - rtclog::StreamConfig, etc. -
  // is kept separate.
  enum class Type : uint32_t {
    AlrStateEvent,
    RouteChangeEvent,
    RemoteEstimateEvent,
    AudioNetworkAdaptation,
    AudioPlayout,
    AudioReceiveStreamConfig,
    AudioSendStreamConfig,
    BweUpdateDelayBased,
    BweUpdateLossBased,
    DtlsTransportState,
    DtlsWritableState,
    IceCandidatePairConfig,
    IceCandidatePairEvent,
    ProbeClusterCreated,
    ProbeResultFailure,
    ProbeResultSuccess,
    RtcpPacketIncoming,
    RtcpPacketOutgoing,
    RtpPacketIncoming,
    RtpPacketOutgoing,
    VideoReceiveStreamConfig,
    VideoSendStreamConfig,
    GenericPacketSent,
    GenericPacketReceived,
    GenericAckReceived,
    FrameDecoded,
    NetEqSetMinimumDelay,
    BeginV3Log = 0x2501580,
    EndV3Log = 0x2501581,
    FakeEvent,  // For unit testing.
  };

  RtcEvent();
  virtual ~RtcEvent() = default;

  virtual Type GetType() const = 0;

  virtual bool IsConfigEvent() const = 0;

  // Events are grouped by Type before being encoded.
  // Optionally, `GetGroupKey` can be overloaded to group the
  // events by a secondary key (in addition to the event type.)
  // This can, in some cases, improve compression efficiency
  // e.g. by grouping events by SSRC.
  virtual uint32_t GetGroupKey() const { return 0; }

  int64_t timestamp_ms() const { return timestamp_us_ / 1000; }
  int64_t timestamp_us() const { return timestamp_us_; }

 protected:
  explicit RtcEvent(int64_t timestamp_us) : timestamp_us_(timestamp_us) {}

  const int64_t timestamp_us_;
};

}  // namespace webrtc

#endif  // API_RTC_EVENT_LOG_RTC_EVENT_H_
