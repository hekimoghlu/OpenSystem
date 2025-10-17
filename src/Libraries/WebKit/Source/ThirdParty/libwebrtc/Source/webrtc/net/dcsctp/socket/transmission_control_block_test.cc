/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 31, 2024.
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
#include "net/dcsctp/socket/transmission_control_block.h"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <type_traits>
#include <vector>

#include "api/array_view.h"
#include "api/task_queue/task_queue_base.h"
#include "net/dcsctp/common/handover_testing.h"
#include "net/dcsctp/common/internal_types.h"
#include "net/dcsctp/packet/chunk/reconfig_chunk.h"
#include "net/dcsctp/packet/parameter/incoming_ssn_reset_request_parameter.h"
#include "net/dcsctp/packet/parameter/outgoing_ssn_reset_request_parameter.h"
#include "net/dcsctp/packet/parameter/parameter.h"
#include "net/dcsctp/packet/parameter/reconfiguration_response_parameter.h"
#include "net/dcsctp/public/dcsctp_message.h"
#include "net/dcsctp/rx/data_tracker.h"
#include "net/dcsctp/rx/reassembly_queue.h"
#include "net/dcsctp/socket/capabilities.h"
#include "net/dcsctp/socket/mock_context.h"
#include "net/dcsctp/socket/mock_dcsctp_socket_callbacks.h"
#include "net/dcsctp/testing/data_generator.h"
#include "net/dcsctp/testing/testing_macros.h"
#include "net/dcsctp/timer/timer.h"
#include "net/dcsctp/tx/mock_send_queue.h"
#include "net/dcsctp/tx/retransmission_queue.h"
#include "rtc_base/gunit.h"
#include "test/gmock.h"

namespace dcsctp {
namespace {
using ::testing::Return;
using ::testing::StrictMock;

constexpr VerificationTag kMyVerificationTag = VerificationTag(123);
constexpr VerificationTag kPeerVerificationTag = VerificationTag(456);
constexpr TSN kMyInitialTsn = TSN(10);
constexpr TSN kPeerInitialTsn = TSN(1000);
constexpr size_t kArwnd = 65536;
constexpr TieTag kTieTag = TieTag(12345678);

class TransmissionControlBlockTest : public testing::Test {
 protected:
  TransmissionControlBlockTest()
      : sender_(callbacks_, on_send_fn_.AsStdFunction()),
        timer_manager_([this](webrtc::TaskQueueBase::DelayPrecision precision) {
          return callbacks_.CreateTimeout(precision);
        }) {}

  DcSctpOptions options_;
  Capabilities capabilities_;
  StrictMock<MockDcSctpSocketCallbacks> callbacks_;
  StrictMock<MockSendQueue> send_queue_;
  testing::MockFunction<void(rtc::ArrayView<const uint8_t>, SendPacketStatus)>
      on_send_fn_;
  testing::MockFunction<bool()> on_connection_established;
  PacketSender sender_;
  TimerManager timer_manager_;
};

TEST_F(TransmissionControlBlockTest, LogsBasicInfoInToString) {
  EXPECT_CALL(send_queue_, EnableMessageInterleaving);

  capabilities_.negotiated_maximum_incoming_streams = 1000;
  capabilities_.negotiated_maximum_outgoing_streams = 2000;
  TransmissionControlBlock tcb(
      timer_manager_, "log: ", options_, capabilities_, callbacks_, send_queue_,
      kMyVerificationTag, kMyInitialTsn, kPeerVerificationTag, kPeerInitialTsn,
      kArwnd, kTieTag, sender_, on_connection_established.AsStdFunction());

  EXPECT_EQ(tcb.ToString(),
            "verification_tag=000001c8, last_cumulative_ack=999, capabilities= "
            "max_in=1000 max_out=2000");
}

TEST_F(TransmissionControlBlockTest, LogsAllCapabilitiesInToSring) {
  EXPECT_CALL(send_queue_, EnableMessageInterleaving);

  capabilities_.negotiated_maximum_incoming_streams = 1000;
  capabilities_.negotiated_maximum_outgoing_streams = 2000;
  capabilities_.message_interleaving = true;
  capabilities_.partial_reliability = true;
  capabilities_.zero_checksum = true;
  capabilities_.reconfig = true;

  TransmissionControlBlock tcb(
      timer_manager_, "log: ", options_, capabilities_, callbacks_, send_queue_,
      kMyVerificationTag, kMyInitialTsn, kPeerVerificationTag, kPeerInitialTsn,
      kArwnd, kTieTag, sender_, on_connection_established.AsStdFunction());

  EXPECT_EQ(
      tcb.ToString(),
      "verification_tag=000001c8, last_cumulative_ack=999, "
      "capabilities=PR,IL,Reconfig,ZeroChecksum, max_in=1000 max_out=2000");
}

TEST_F(TransmissionControlBlockTest, IsInitiallyHandoverReady) {
  EXPECT_CALL(send_queue_, EnableMessageInterleaving);
  EXPECT_CALL(send_queue_, HasStreamsReadyToBeReset).WillOnce(Return(false));

  TransmissionControlBlock tcb(
      timer_manager_, "log: ", options_, capabilities_, callbacks_, send_queue_,
      kMyVerificationTag, kMyInitialTsn, kPeerVerificationTag, kPeerInitialTsn,
      kArwnd, kTieTag, sender_, on_connection_established.AsStdFunction());

  EXPECT_TRUE(tcb.GetHandoverReadiness().IsReady());
}
}  // namespace
}  // namespace dcsctp
