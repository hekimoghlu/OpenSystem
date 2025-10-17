/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 26, 2025.
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
#include "modules/congestion_controller/include/receive_side_congestion_controller.h"

#include <algorithm>
#include <memory>
#include <utility>

#include "absl/base/nullability.h"
#include "api/environment/environment.h"
#include "api/media_types.h"
#include "api/sequence_checker.h"
#include "api/units/data_rate.h"
#include "modules/remote_bitrate_estimator/congestion_control_feedback_generator.h"
#include "modules/remote_bitrate_estimator/remote_bitrate_estimator_abs_send_time.h"
#include "modules/remote_bitrate_estimator/remote_bitrate_estimator_single_stream.h"
#include "modules/remote_bitrate_estimator/transport_sequence_number_feedback_generator.h"
#include "modules/rtp_rtcp/source/rtp_header_extensions.h"
#include "rtc_base/logging.h"

namespace webrtc {

namespace {
static const uint32_t kTimeOffsetSwitchThreshold = 30;
}  // namespace

void ReceiveSideCongestionController::OnRttUpdate(int64_t avg_rtt_ms,
                                                  int64_t max_rtt_ms) {
  MutexLock lock(&mutex_);
  rbe_->OnRttUpdate(avg_rtt_ms, max_rtt_ms);
}

void ReceiveSideCongestionController::RemoveStream(uint32_t ssrc) {
  MutexLock lock(&mutex_);
  rbe_->RemoveStream(ssrc);
}

DataRate ReceiveSideCongestionController::LatestReceiveSideEstimate() const {
  MutexLock lock(&mutex_);
  return rbe_->LatestEstimate();
}

void ReceiveSideCongestionController::PickEstimator(
    bool has_absolute_send_time) {
  if (has_absolute_send_time) {
    // If we see AST in header, switch RBE strategy immediately.
    if (!using_absolute_send_time_) {
      RTC_LOG(LS_INFO)
          << "WrappingBitrateEstimator: Switching to absolute send time RBE.";
      using_absolute_send_time_ = true;
      rbe_ = std::make_unique<RemoteBitrateEstimatorAbsSendTime>(
          env_, &remb_throttler_);
    }
    packets_since_absolute_send_time_ = 0;
  } else {
    // When we don't see AST, wait for a few packets before going back to TOF.
    if (using_absolute_send_time_) {
      ++packets_since_absolute_send_time_;
      if (packets_since_absolute_send_time_ >= kTimeOffsetSwitchThreshold) {
        RTC_LOG(LS_INFO)
            << "WrappingBitrateEstimator: Switching to transmission "
               "time offset RBE.";
        using_absolute_send_time_ = false;
        rbe_ = std::make_unique<RemoteBitrateEstimatorSingleStream>(
            env_, &remb_throttler_);
      }
    }
  }
}

ReceiveSideCongestionController::ReceiveSideCongestionController(
    const Environment& env,
    TransportSequenceNumberFeedbackGenenerator::RtcpSender feedback_sender,
    RembThrottler::RembSender remb_sender,
    absl::Nullable<NetworkStateEstimator*> network_state_estimator)
    : env_(env),
      remb_throttler_(std::move(remb_sender), &env_.clock()),
      transport_sequence_number_feedback_generator_(feedback_sender,
                                                    network_state_estimator),
      congestion_control_feedback_generator_(env, feedback_sender),
      rbe_(std::make_unique<RemoteBitrateEstimatorSingleStream>(
          env_,
          &remb_throttler_)),
      using_absolute_send_time_(false),
      packets_since_absolute_send_time_(0) {
  FieldTrialParameter<bool> force_send_rfc8888_feedback("force_send", false);
  ParseFieldTrial(
      {&force_send_rfc8888_feedback},
      env.field_trials().Lookup("WebRTC-RFC8888CongestionControlFeedback"));
  if (force_send_rfc8888_feedback) {
    EnablSendCongestionControlFeedbackAccordingToRfc8888();
  }
}

void ReceiveSideCongestionController::
    EnablSendCongestionControlFeedbackAccordingToRfc8888() {
  RTC_DCHECK_RUN_ON(&sequence_checker_);
  send_rfc8888_congestion_feedback_ = true;
}

void ReceiveSideCongestionController::OnReceivedPacket(
    const RtpPacketReceived& packet,
    MediaType media_type) {
  bool has_transport_sequence_number =
      packet.HasExtension<TransportSequenceNumber>() ||
      packet.HasExtension<TransportSequenceNumberV2>();
  if (send_rfc8888_congestion_feedback_) {
    RTC_DCHECK_RUN_ON(&sequence_checker_);
    congestion_control_feedback_generator_.OnReceivedPacket(packet);
    // TODO(https://bugs.webrtc.org/374197376): Utilize RFC 8888 feedback, which
    // provides comprehensive details similar to transport-cc. To ensure a
    // smooth transition, we will continue using transport sequence number
    // feedback temporarily. Once validation is complete, we will fully
    // transition to using RFC 8888 feedback exclusively.
    if (has_transport_sequence_number) {
      transport_sequence_number_feedback_generator_.OnReceivedPacket(packet);
    }
    return;
  }
  if (media_type == MediaType::AUDIO && !has_transport_sequence_number) {
    // For audio, we only support send side BWE.
    return;
  }

  if (has_transport_sequence_number) {
    // Send-side BWE.
    transport_sequence_number_feedback_generator_.OnReceivedPacket(packet);
  } else {
    // Receive-side BWE.
    MutexLock lock(&mutex_);
    PickEstimator(packet.HasExtension<AbsoluteSendTime>());
    rbe_->IncomingPacket(packet);
  }
}

void ReceiveSideCongestionController::OnBitrateChanged(int bitrate_bps) {
  RTC_DCHECK_RUN_ON(&sequence_checker_);
  DataRate send_bandwidth_estimate = DataRate::BitsPerSec(bitrate_bps);
  transport_sequence_number_feedback_generator_.OnSendBandwidthEstimateChanged(
      send_bandwidth_estimate);
  congestion_control_feedback_generator_.OnSendBandwidthEstimateChanged(
      send_bandwidth_estimate);
}

TimeDelta ReceiveSideCongestionController::MaybeProcess() {
  Timestamp now = env_.clock().CurrentTime();
  if (send_rfc8888_congestion_feedback_) {
    RTC_DCHECK_RUN_ON(&sequence_checker_);
    TimeDelta time_until_cc_rep =
        congestion_control_feedback_generator_.Process(now);
    TimeDelta time_until_rep =
        transport_sequence_number_feedback_generator_.Process(now);
    TimeDelta time_until = std::min(time_until_cc_rep, time_until_rep);
    return std::max(time_until, TimeDelta::Zero());
  }
  mutex_.Lock();
  TimeDelta time_until_rbe = rbe_->Process();
  mutex_.Unlock();
  TimeDelta time_until_rep =
      transport_sequence_number_feedback_generator_.Process(now);
  TimeDelta time_until = std::min(time_until_rbe, time_until_rep);
  return std::max(time_until, TimeDelta::Zero());
}

void ReceiveSideCongestionController::SetMaxDesiredReceiveBitrate(
    DataRate bitrate) {
  remb_throttler_.SetMaxDesiredReceiveBitrate(bitrate);
}

void ReceiveSideCongestionController::SetTransportOverhead(
    DataSize overhead_per_packet) {
  RTC_DCHECK_RUN_ON(&sequence_checker_);
  transport_sequence_number_feedback_generator_.SetTransportOverhead(
      overhead_per_packet);
  congestion_control_feedback_generator_.SetTransportOverhead(
      overhead_per_packet);
}

}  // namespace webrtc
