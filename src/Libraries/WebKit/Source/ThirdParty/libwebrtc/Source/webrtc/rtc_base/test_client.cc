/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 9, 2024.
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
#include "rtc_base/test_client.h"

#include <string.h>

#include <memory>
#include <utility>

#include "api/units/timestamp.h"
#include "rtc_base/gunit.h"
#include "rtc_base/network/received_packet.h"
#include "rtc_base/thread.h"
#include "rtc_base/time_utils.h"

namespace rtc {

// DESIGN: Each packet received is put it into a list of packets.
//         Callers can retrieve received packets from any thread by calling
//         NextPacket.

TestClient::TestClient(std::unique_ptr<AsyncPacketSocket> socket)
    : TestClient(std::move(socket), nullptr) {}

TestClient::TestClient(std::unique_ptr<AsyncPacketSocket> socket,
                       ThreadProcessingFakeClock* fake_clock)
    : fake_clock_(fake_clock), socket_(std::move(socket)) {
  socket_->RegisterReceivedPacketCallback(
      [&](rtc::AsyncPacketSocket* socket, const rtc::ReceivedPacket& packet) {
        OnPacket(socket, packet);
      });
  socket_->SignalReadyToSend.connect(this, &TestClient::OnReadyToSend);
}

TestClient::~TestClient() {}

bool TestClient::CheckConnState(AsyncPacketSocket::State state) {
  // Wait for our timeout value until the socket reaches the desired state.
  int64_t end = TimeAfter(kTimeoutMs);
  while (socket_->GetState() != state && TimeUntil(end) > 0) {
    AdvanceTime(1);
  }
  return (socket_->GetState() == state);
}

int TestClient::Send(const char* buf, size_t size) {
  rtc::PacketOptions options;
  return socket_->Send(buf, size, options);
}

int TestClient::SendTo(const char* buf,
                       size_t size,
                       const SocketAddress& dest) {
  rtc::PacketOptions options;
  return socket_->SendTo(buf, size, dest, options);
}

std::unique_ptr<TestClient::Packet> TestClient::NextPacket(int timeout_ms) {
  // If no packets are currently available, we go into a get/dispatch loop for
  // at most timeout_ms.  If, during the loop, a packet arrives, then we can
  // stop early and return it.

  // Note that the case where no packet arrives is important.  We often want to
  // test that a packet does not arrive.

  // Note also that we only try to pump our current thread's message queue.
  // Pumping another thread's queue could lead to messages being dispatched from
  // the wrong thread to non-thread-safe objects.

  int64_t end = TimeAfter(timeout_ms);
  while (TimeUntil(end) > 0) {
    {
      webrtc::MutexLock lock(&mutex_);
      if (packets_.size() != 0) {
        break;
      }
    }
    AdvanceTime(1);
  }

  // Return the first packet placed in the queue.
  std::unique_ptr<Packet> packet;
  webrtc::MutexLock lock(&mutex_);
  if (packets_.size() > 0) {
    packet = std::move(packets_.front());
    packets_.erase(packets_.begin());
  }

  return packet;
}

bool TestClient::CheckNextPacket(const char* buf,
                                 size_t size,
                                 SocketAddress* addr) {
  bool res = false;
  std::unique_ptr<Packet> packet = NextPacket(kTimeoutMs);
  if (packet) {
    res = (packet->buf.size() == size &&
           memcmp(packet->buf.data(), buf, size) == 0 &&
           CheckTimestamp(packet->packet_time));
    if (addr)
      *addr = packet->addr;
  }
  return res;
}

bool TestClient::CheckTimestamp(
    std::optional<webrtc::Timestamp> packet_timestamp) {
  bool res = true;
  if (!packet_timestamp) {
    res = false;
  }
  if (prev_packet_timestamp_) {
    if (packet_timestamp < prev_packet_timestamp_) {
      res = false;
    }
  }
  prev_packet_timestamp_ = packet_timestamp;
  return res;
}

void TestClient::AdvanceTime(int ms) {
  // If the test is using a fake clock, we must advance the fake clock to
  // advance time. Otherwise, ProcessMessages will work.
  if (fake_clock_) {
    SIMULATED_WAIT(false, ms, *fake_clock_);
  } else {
    Thread::Current()->ProcessMessages(1);
  }
}

bool TestClient::CheckNoPacket() {
  return NextPacket(kNoPacketTimeoutMs) == nullptr;
}

int TestClient::GetError() {
  return socket_->GetError();
}

int TestClient::SetOption(Socket::Option opt, int value) {
  return socket_->SetOption(opt, value);
}

void TestClient::OnPacket(AsyncPacketSocket* socket,
                          const rtc::ReceivedPacket& received_packet) {
  webrtc::MutexLock lock(&mutex_);
  packets_.push_back(std::make_unique<Packet>(received_packet));
}

void TestClient::OnReadyToSend(AsyncPacketSocket* socket) {
  ++ready_to_send_count_;
}

TestClient::Packet::Packet(const rtc::ReceivedPacket& received_packet)
    : addr(received_packet.source_address()),
      // Copy received_packet payload to a buffer owned by Packet.
      buf(received_packet.payload().data(), received_packet.payload().size()),
      packet_time(received_packet.arrival_time()) {}

TestClient::Packet::Packet(const Packet& p)
    : addr(p.addr),
      buf(p.buf.data(), p.buf.size()),
      packet_time(p.packet_time) {}

}  // namespace rtc
