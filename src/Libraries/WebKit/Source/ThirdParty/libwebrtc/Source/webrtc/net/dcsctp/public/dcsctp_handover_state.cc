/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 23, 2023.
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
#include "net/dcsctp/public/dcsctp_handover_state.h"

#include <string>

#include "absl/strings/string_view.h"

namespace dcsctp {
namespace {
constexpr absl::string_view HandoverUnreadinessReasonToString(
    HandoverUnreadinessReason reason) {
  switch (reason) {
    case HandoverUnreadinessReason::kWrongConnectionState:
      return "WRONG_CONNECTION_STATE";
    case HandoverUnreadinessReason::kSendQueueNotEmpty:
      return "SEND_QUEUE_NOT_EMPTY";
    case HandoverUnreadinessReason::kDataTrackerTsnBlocksPending:
      return "DATA_TRACKER_TSN_BLOCKS_PENDING";
    case HandoverUnreadinessReason::kReassemblyQueueDeliveredTSNsGap:
      return "REASSEMBLY_QUEUE_DELIVERED_TSN_GAP";
    case HandoverUnreadinessReason::kStreamResetDeferred:
      return "STREAM_RESET_DEFERRED";
    case HandoverUnreadinessReason::kOrderedStreamHasUnassembledChunks:
      return "ORDERED_STREAM_HAS_UNASSEMBLED_CHUNKS";
    case HandoverUnreadinessReason::kUnorderedStreamHasUnassembledChunks:
      return "UNORDERED_STREAM_HAS_UNASSEMBLED_CHUNKS";
    case HandoverUnreadinessReason::kRetransmissionQueueOutstandingData:
      return "RETRANSMISSION_QUEUE_OUTSTANDING_DATA";
    case HandoverUnreadinessReason::kRetransmissionQueueFastRecovery:
      return "RETRANSMISSION_QUEUE_FAST_RECOVERY";
    case HandoverUnreadinessReason::kRetransmissionQueueNotEmpty:
      return "RETRANSMISSION_QUEUE_NOT_EMPTY";
    case HandoverUnreadinessReason::kPendingStreamReset:
      return "PENDING_STREAM_RESET";
    case HandoverUnreadinessReason::kPendingStreamResetRequest:
      return "PENDING_STREAM_RESET_REQUEST";
  }
}
}  // namespace

std::string HandoverReadinessStatus::ToString() const {
  std::string result;
  for (uint32_t bit = 1;
       bit <= static_cast<uint32_t>(HandoverUnreadinessReason::kMax);
       bit *= 2) {
    auto flag = static_cast<HandoverUnreadinessReason>(bit);
    if (Contains(flag)) {
      if (!result.empty()) {
        result.append(",");
      }
      absl::string_view s = HandoverUnreadinessReasonToString(flag);
      result.append(s.data(), s.size());
    }
  }
  if (result.empty()) {
    result = "READY";
  }
  return result;
}
}  // namespace dcsctp
