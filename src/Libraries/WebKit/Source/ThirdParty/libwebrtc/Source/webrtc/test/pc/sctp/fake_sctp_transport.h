/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 10, 2023.
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
#ifndef TEST_PC_SCTP_FAKE_SCTP_TRANSPORT_H_
#define TEST_PC_SCTP_FAKE_SCTP_TRANSPORT_H_

#include <memory>

#include "api/environment/environment.h"
#include "api/priority.h"
#include "api/transport/sctp_transport_factory_interface.h"
#include "media/sctp/sctp_transport_internal.h"

// Used for tests in this file to verify that PeerConnection responds to signals
// from the SctpTransport correctly, and calls Start with the correct
// local/remote ports.
class FakeSctpTransport : public cricket::SctpTransportInternal {
 public:
  void SetOnConnectedCallback(std::function<void()> callback) override {}
  void SetDataChannelSink(webrtc::DataChannelSink* sink) override {}
  void SetDtlsTransport(rtc::PacketTransportInternal* transport) override {}
  bool Start(int local_port, int remote_port, int max_message_size) override {
    local_port_.emplace(local_port);
    remote_port_.emplace(remote_port);
    max_message_size_ = max_message_size;
    return true;
  }
  bool OpenStream(int sid, webrtc::PriorityValue priority) override {
    return true;
  }
  bool ResetStream(int sid) override { return true; }
  webrtc::RTCError SendData(int sid,
                            const webrtc::SendDataParams& params,
                            const rtc::CopyOnWriteBuffer& payload) override {
    return webrtc::RTCError::OK();
  }
  bool ReadyToSendData() override { return true; }
  void set_debug_name_for_testing(const char* debug_name) override {}

  int max_message_size() const override { return max_message_size_; }
  std::optional<int> max_outbound_streams() const override {
    return std::nullopt;
  }
  std::optional<int> max_inbound_streams() const override {
    return std::nullopt;
  }
  size_t buffered_amount(int sid) const override { return 0; }
  size_t buffered_amount_low_threshold(int sid) const override { return 0; }
  void SetBufferedAmountLowThreshold(int sid, size_t bytes) override {}
  int local_port() const {
    RTC_DCHECK(local_port_);
    return *local_port_;
  }
  int remote_port() const {
    RTC_DCHECK(remote_port_);
    return *remote_port_;
  }

 private:
  std::optional<int> local_port_;
  std::optional<int> remote_port_;
  int max_message_size_;
};

class FakeSctpTransportFactory : public webrtc::SctpTransportFactoryInterface {
 public:
  std::unique_ptr<cricket::SctpTransportInternal> CreateSctpTransport(
      const webrtc::Environment& env,
      rtc::PacketTransportInternal*) override {
    last_fake_sctp_transport_ = new FakeSctpTransport();
    return std::unique_ptr<cricket::SctpTransportInternal>(
        last_fake_sctp_transport_);
  }

  FakeSctpTransport* last_fake_sctp_transport() {
    return last_fake_sctp_transport_;
  }

 private:
  FakeSctpTransport* last_fake_sctp_transport_ = nullptr;
};

#endif  // TEST_PC_SCTP_FAKE_SCTP_TRANSPORT_H_
