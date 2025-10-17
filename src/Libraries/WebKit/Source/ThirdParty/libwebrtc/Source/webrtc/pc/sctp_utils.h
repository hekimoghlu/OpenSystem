/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 18, 2022.
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
#ifndef PC_SCTP_UTILS_H_
#define PC_SCTP_UTILS_H_

#include <string>

#include "api/data_channel_interface.h"
#include "api/priority.h"
#include "api/transport/data_channel_transport_interface.h"
#include "media/base/media_channel.h"
#include "media/sctp/sctp_transport_internal.h"
#include "net/dcsctp/public/types.h"
#include "rtc_base/copy_on_write_buffer.h"
#include "rtc_base/ssl_stream_adapter.h"  // For SSLRole

namespace rtc {
class CopyOnWriteBuffer;
}  // namespace rtc

namespace webrtc {
struct DataChannelInit;

// Wraps the `uint16_t` sctp data channel stream id value and does range
// checking. The class interface is `int` based to ease with DataChannelInit
// compatibility and types used in `DataChannelController`'s interface. Going
// forward, `int` compatibility won't be needed and we can either just use
// this class or the internal dcsctp::StreamID type.
class StreamId {
 public:
  StreamId() = default;
  explicit StreamId(uint16_t id) : id_(id) {}
  StreamId(const StreamId& sid) = default;
  StreamId& operator=(const StreamId& sid) = default;
  // Provided for compatibility with existing code that hasn't been updated
  // to use `StreamId` directly. New code should not use 'int' for the stream
  // id but rather `StreamId` directly.
  int stream_id_int() const { return static_cast<int>(id_.value()); }

  bool operator==(const StreamId& sid) const { return id_ == sid.id_; }
  bool operator<(const StreamId& sid) const { return id_ < sid.id_; }
  bool operator!=(const StreamId& sid) const { return !(operator==(sid)); }

 private:
  dcsctp::StreamID id_;
};

// Read the message type and return true if it's an OPEN message.
bool IsOpenMessage(const rtc::CopyOnWriteBuffer& payload);

bool ParseDataChannelOpenMessage(const rtc::CopyOnWriteBuffer& payload,
                                 std::string* label,
                                 DataChannelInit* config);

bool ParseDataChannelOpenAckMessage(const rtc::CopyOnWriteBuffer& payload);

bool WriteDataChannelOpenMessage(const std::string& label,
                                 const std::string& protocol,
                                 std::optional<PriorityValue> priority,
                                 bool ordered,
                                 std::optional<int> max_retransmits,
                                 std::optional<int> max_retransmit_time,
                                 rtc::CopyOnWriteBuffer* payload);
bool WriteDataChannelOpenMessage(const std::string& label,
                                 const DataChannelInit& config,
                                 rtc::CopyOnWriteBuffer* payload);
void WriteDataChannelOpenAckMessage(rtc::CopyOnWriteBuffer* payload);

}  // namespace webrtc

#endif  // PC_SCTP_UTILS_H_
