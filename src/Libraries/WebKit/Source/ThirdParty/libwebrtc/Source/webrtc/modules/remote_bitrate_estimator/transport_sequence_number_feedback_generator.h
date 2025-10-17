/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 31, 2021.
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
#ifndef MODULES_REMOTE_BITRATE_ESTIMATOR_TRANSPORT_SEQUENCE_NUMBER_FEEDBACK_GENERATOR_H_
#define MODULES_REMOTE_BITRATE_ESTIMATOR_TRANSPORT_SEQUENCE_NUMBER_FEEDBACK_GENERATOR_H_

#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

#include "api/field_trials_view.h"
#include "api/rtp_headers.h"
#include "api/transport/network_control.h"
#include "api/units/data_rate.h"
#include "api/units/data_size.h"
#include "api/units/time_delta.h"
#include "api/units/timestamp.h"
#include "modules/remote_bitrate_estimator/packet_arrival_map.h"
#include "modules/remote_bitrate_estimator/rtp_transport_feedback_generator.h"
#include "modules/rtp_rtcp/source/rtcp_packet.h"
#include "modules/rtp_rtcp/source/rtcp_packet/transport_feedback.h"
#include "modules/rtp_rtcp/source/rtp_packet_received.h"
#include "rtc_base/numerics/sequence_number_unwrapper.h"
#include "rtc_base/synchronization/mutex.h"

namespace webrtc {

// Class used when send-side BWE is enabled.
// The class is responsible for generating RTCP feedback packets based on
// incoming media packets. Incoming packets must have a transport sequence
// number, Ie. either the extension
// http://www.ietf.org/id/draft-holmer-rmcat-transport-wide-cc-extensions-01 or
// http://www.webrtc.org/experiments/rtp-hdrext/transport-wide-cc-02 must be
// used.
class TransportSequenceNumberFeedbackGenenerator
    : public RtpTransportFeedbackGenerator {
 public:
  TransportSequenceNumberFeedbackGenenerator(
      RtpTransportFeedbackGenerator::RtcpSender feedback_sender,
      NetworkStateEstimator* network_state_estimator);
  ~TransportSequenceNumberFeedbackGenenerator();

  void OnReceivedPacket(const RtpPacketReceived& packet) override;
  void OnSendBandwidthEstimateChanged(DataRate estimate) override;

  TimeDelta Process(Timestamp now) override;

  void SetTransportOverhead(DataSize overhead_per_packet) override;

 private:
  void MaybeCullOldPackets(int64_t sequence_number, Timestamp arrival_time)
      RTC_EXCLUSIVE_LOCKS_REQUIRED(&lock_);
  void SendPeriodicFeedbacks() RTC_EXCLUSIVE_LOCKS_REQUIRED(&lock_);
  void SendFeedbackOnRequest(int64_t sequence_number,
                             const FeedbackRequest& feedback_request)
      RTC_EXCLUSIVE_LOCKS_REQUIRED(&lock_);

  // Returns a Transport Feedback packet with information about as many
  // packets that has been received between [`begin_sequence_number_incl`,
  // `end_sequence_number_excl`) that can fit in it. If `is_periodic_update`,
  // this represents sending a periodic feedback message, which will make it
  // update the `periodic_window_start_seq_` variable with the first packet
  // that was not included in the feedback packet, so that the next update can
  // continue from that sequence number.
  //
  // If no incoming packets were added, nullptr is returned.
  //
  // `include_timestamps` decide if the returned TransportFeedback should
  // include timestamps.
  std::unique_ptr<rtcp::TransportFeedback> MaybeBuildFeedbackPacket(
      bool include_timestamps,
      int64_t begin_sequence_number_inclusive,
      int64_t end_sequence_number_exclusive,
      bool is_periodic_update) RTC_EXCLUSIVE_LOCKS_REQUIRED(&lock_);

  const RtcpSender feedback_sender_;
  Timestamp last_process_time_;

  Mutex lock_;
  //  `network_state_estimator_` may be null.
  NetworkStateEstimator* const network_state_estimator_
      RTC_PT_GUARDED_BY(&lock_);
  uint32_t media_ssrc_ RTC_GUARDED_BY(&lock_);
  uint8_t feedback_packet_count_ RTC_GUARDED_BY(&lock_);
  SeqNumUnwrapper<uint16_t> unwrapper_ RTC_GUARDED_BY(&lock_);
  DataSize packet_overhead_ RTC_GUARDED_BY(&lock_);

  // The next sequence number that should be the start sequence number during
  // periodic reporting. Will be std::nullopt before the first seen packet.
  std::optional<int64_t> periodic_window_start_seq_ RTC_GUARDED_BY(&lock_);

  // Packet arrival times, by sequence number.
  PacketArrivalTimeMap packet_arrival_times_ RTC_GUARDED_BY(&lock_);

  TimeDelta send_interval_ RTC_GUARDED_BY(&lock_);
  bool send_periodic_feedback_ RTC_GUARDED_BY(&lock_);

  // Unwraps absolute send times.
  uint32_t previous_abs_send_time_ RTC_GUARDED_BY(&lock_);
  Timestamp abs_send_timestamp_ RTC_GUARDED_BY(&lock_);
  Timestamp last_arrival_time_with_abs_send_time_ RTC_GUARDED_BY(&lock_);
};

}  // namespace webrtc

#endif  // MODULES_REMOTE_BITRATE_ESTIMATOR_TRANSPORT_SEQUENCE_NUMBER_FEEDBACK_GENERATOR_H_
