/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 14, 2022.
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
#ifndef NET_DCSCTP_SOCKET_CALLBACK_DEFERRER_H_
#define NET_DCSCTP_SOCKET_CALLBACK_DEFERRER_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/variant.h"
#include "api/array_view.h"
#include "api/ref_counted_base.h"
#include "api/scoped_refptr.h"
#include "api/task_queue/task_queue_base.h"
#include "net/dcsctp/public/dcsctp_message.h"
#include "net/dcsctp/public/dcsctp_socket.h"

namespace dcsctp {
// Defers callbacks until they can be safely triggered.
//
// There are a lot of callbacks from the dcSCTP library to the client,
// such as when messages are received or streams are closed. When the client
// receives these callbacks, the client is expected to be able to call into the
// library - from within the callback. For example, sending a reply message when
// a certain SCTP message has been received, or to reconnect when the connection
// was closed for any reason. This means that the dcSCTP library must always be
// in a consistent and stable state when these callbacks are delivered, and to
// ensure that's the case, callbacks are not immediately delivered from where
// they originate, but instead queued (deferred) by this class. At the end of
// any public API method that may result in callbacks, they are triggered and
// then delivered.
//
// There are a number of exceptions, which is clearly annotated in the API.
class CallbackDeferrer : public DcSctpSocketCallbacks {
 public:
  class ScopedDeferrer {
   public:
    explicit ScopedDeferrer(CallbackDeferrer& callback_deferrer)
        : callback_deferrer_(callback_deferrer) {
      callback_deferrer_.Prepare();
    }

    ~ScopedDeferrer() { callback_deferrer_.TriggerDeferred(); }

   private:
    CallbackDeferrer& callback_deferrer_;
  };

  explicit CallbackDeferrer(DcSctpSocketCallbacks& underlying)
      : underlying_(underlying) {}

  // Implementation of DcSctpSocketCallbacks
  SendPacketStatus SendPacketWithStatus(
      rtc::ArrayView<const uint8_t> data) override;
  std::unique_ptr<Timeout> CreateTimeout(
      webrtc::TaskQueueBase::DelayPrecision precision) override;
  TimeMs TimeMillis() override;
  webrtc::Timestamp Now() override { return underlying_.Now(); }
  uint32_t GetRandomInt(uint32_t low, uint32_t high) override;
  void OnMessageReceived(DcSctpMessage message) override;
  void OnError(ErrorKind error, absl::string_view message) override;
  void OnAborted(ErrorKind error, absl::string_view message) override;
  void OnConnected() override;
  void OnClosed() override;
  void OnConnectionRestarted() override;
  void OnStreamsResetFailed(rtc::ArrayView<const StreamID> outgoing_streams,
                            absl::string_view reason) override;
  void OnStreamsResetPerformed(
      rtc::ArrayView<const StreamID> outgoing_streams) override;
  void OnIncomingStreamsReset(
      rtc::ArrayView<const StreamID> incoming_streams) override;
  void OnBufferedAmountLow(StreamID stream_id) override;
  void OnTotalBufferedAmountLow() override;

  void OnLifecycleMessageExpired(LifecycleId lifecycle_id,
                                 bool maybe_delivered) override;
  void OnLifecycleMessageFullySent(LifecycleId lifecycle_id) override;
  void OnLifecycleMessageDelivered(LifecycleId lifecycle_id) override;
  void OnLifecycleEnd(LifecycleId lifecycle_id) override;

 private:
  struct Error {
    ErrorKind error;
    std::string message;
  };
  struct StreamReset {
    std::vector<StreamID> streams;
    std::string message;
  };
  // Use a pre-sized variant for storage to avoid double heap allocation. This
  // variant can hold all cases of stored data.
  using CallbackData = absl::
      variant<absl::monostate, DcSctpMessage, Error, StreamReset, StreamID>;
  using Callback = void (*)(CallbackData, DcSctpSocketCallbacks&);

  void Prepare();
  void TriggerDeferred();

  DcSctpSocketCallbacks& underlying_;
  bool prepared_ = false;
  std::vector<std::pair<Callback, CallbackData>> deferred_;
};
}  // namespace dcsctp

#endif  // NET_DCSCTP_SOCKET_CALLBACK_DEFERRER_H_
