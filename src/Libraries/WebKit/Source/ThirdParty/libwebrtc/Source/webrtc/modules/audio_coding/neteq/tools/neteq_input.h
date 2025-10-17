/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 16, 2023.
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
#ifndef MODULES_AUDIO_CODING_NETEQ_TOOLS_NETEQ_INPUT_H_
#define MODULES_AUDIO_CODING_NETEQ_TOOLS_NETEQ_INPUT_H_

#include <algorithm>
#include <memory>
#include <optional>
#include <string>

#include "modules/audio_coding/neteq/tools/packet.h"
#include "modules/audio_coding/neteq/tools/packet_source.h"
#include "rtc_base/buffer.h"

namespace webrtc {
namespace test {

// Interface class for input to the NetEqTest class.
class NetEqInput {
 public:
  struct PacketData {
    PacketData();
    ~PacketData();
    std::string ToString() const;

    RTPHeader header;
    rtc::Buffer payload;
    int64_t time_ms;
  };

  struct SetMinimumDelayInfo {
    SetMinimumDelayInfo(int64_t timestamp_ms_in, int delay_ms_in)
        : timestamp_ms(timestamp_ms_in), delay_ms(delay_ms_in) {}
    int64_t timestamp_ms;
    int delay_ms;
  };

  virtual ~NetEqInput() = default;

  // Returns at what time (in ms) NetEq::InsertPacket should be called next, or
  // empty if the source is out of packets.
  virtual std::optional<int64_t> NextPacketTime() const = 0;

  // Returns at what time (in ms) NetEq::GetAudio should be called next, or
  // empty if no more output events are available.
  virtual std::optional<int64_t> NextOutputEventTime() const = 0;

  // Returns the information related to the next NetEq set minimum delay event
  // if available.
  virtual std::optional<SetMinimumDelayInfo> NextSetMinimumDelayInfo()
      const = 0;

  // Returns the time (in ms) for the next event (packet, output or set minimum
  // delay event) or empty if there are no more events.
  std::optional<int64_t> NextEventTime() const {
    std::optional<int64_t> next_event_time = NextPacketTime();
    const auto next_output_time = NextOutputEventTime();
    // Return the minimum of non-empty `a` and `b`, or empty if both are empty.
    if (next_output_time) {
      next_event_time = next_event_time ? std::min(next_event_time.value(),
                                                   next_output_time.value())
                                        : next_output_time;
    }
    const auto next_neteq_minimum_delay = NextSetMinimumDelayInfo();
    if (next_neteq_minimum_delay) {
      next_event_time =
          next_event_time
              ? std::min(next_event_time.value(),
                         next_neteq_minimum_delay.value().timestamp_ms)
              : next_neteq_minimum_delay.value().timestamp_ms;
    }
    return next_event_time;
  }

  // Returns the next packet to be inserted into NetEq. The packet following the
  // returned one is pre-fetched in the NetEqInput object, such that future
  // calls to NextPacketTime() or NextHeader() will return information from that
  // packet.
  virtual std::unique_ptr<PacketData> PopPacket() = 0;

  // Move to the next output event. This will make NextOutputEventTime() return
  // a new value (potentially the same if several output events share the same
  // time).
  virtual void AdvanceOutputEvent() = 0;

  // Move to the next NetEq set minimum delay. This will make
  // `NextSetMinimumDelayInfo` return a new value.
  virtual void AdvanceSetMinimumDelay() = 0;

  // Returns true if the source has come to an end. An implementation must
  // eventually return true from this method, or the test will end up in an
  // infinite loop.
  virtual bool ended() const = 0;

  // Returns the RTP header for the next packet, i.e., the packet that will be
  // delivered next by PopPacket().
  virtual std::optional<RTPHeader> NextHeader() const = 0;
};

// Wrapper class to impose a time limit on a NetEqInput object, typically
// another time limit than what the object itself provides. For example, an
// input taken from a file can be cut shorter by wrapping it in this class.
class TimeLimitedNetEqInput : public NetEqInput {
 public:
  TimeLimitedNetEqInput(std::unique_ptr<NetEqInput> input, int64_t duration_ms);
  ~TimeLimitedNetEqInput() override;
  std::optional<int64_t> NextPacketTime() const override;
  std::optional<int64_t> NextOutputEventTime() const override;
  std::optional<SetMinimumDelayInfo> NextSetMinimumDelayInfo() const override;
  std::unique_ptr<PacketData> PopPacket() override;
  void AdvanceOutputEvent() override;
  void AdvanceSetMinimumDelay() override;
  bool ended() const override;
  std::optional<RTPHeader> NextHeader() const override;

 private:
  void MaybeSetEnded();

  std::unique_ptr<NetEqInput> input_;
  const std::optional<int64_t> start_time_ms_;
  const int64_t duration_ms_;
  bool ended_ = false;
};

}  // namespace test
}  // namespace webrtc
#endif  // MODULES_AUDIO_CODING_NETEQ_TOOLS_NETEQ_INPUT_H_
