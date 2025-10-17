/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 13, 2023.
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
#include "api/video_codecs/vp8_temporal_layers.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "api/fec_controller_override.h"
#include "api/video_codecs/video_encoder.h"
#include "api/video_codecs/vp8_frame_buffer_controller.h"
#include "api/video_codecs/vp8_frame_config.h"
#include "rtc_base/checks.h"

namespace webrtc {

Vp8TemporalLayers::Vp8TemporalLayers(
    std::vector<std::unique_ptr<Vp8FrameBufferController>>&& controllers,
    FecControllerOverride* fec_controller_override)
    : controllers_(std::move(controllers)) {
  RTC_DCHECK(!controllers_.empty());
  RTC_DCHECK(absl::c_none_of(
      controllers_,
      [](const std::unique_ptr<Vp8FrameBufferController>& controller) {
        return controller.get() == nullptr;
      }));
  if (fec_controller_override) {
    fec_controller_override->SetFecAllowed(true);
  }
}

void Vp8TemporalLayers::SetQpLimits(size_t stream_index,
                                    int min_qp,
                                    int max_qp) {
  RTC_DCHECK_LT(stream_index, controllers_.size());
  return controllers_[stream_index]->SetQpLimits(0, min_qp, max_qp);
}

size_t Vp8TemporalLayers::StreamCount() const {
  return controllers_.size();
}

bool Vp8TemporalLayers::SupportsEncoderFrameDropping(
    size_t stream_index) const {
  RTC_DCHECK_LT(stream_index, controllers_.size());
  return controllers_[stream_index]->SupportsEncoderFrameDropping(0);
}

void Vp8TemporalLayers::OnRatesUpdated(
    size_t stream_index,
    const std::vector<uint32_t>& bitrates_bps,
    int framerate_fps) {
  RTC_DCHECK_LT(stream_index, controllers_.size());
  return controllers_[stream_index]->OnRatesUpdated(0, bitrates_bps,
                                                    framerate_fps);
}

Vp8EncoderConfig Vp8TemporalLayers::UpdateConfiguration(size_t stream_index) {
  RTC_DCHECK_LT(stream_index, controllers_.size());
  return controllers_[stream_index]->UpdateConfiguration(0);
}

Vp8FrameConfig Vp8TemporalLayers::NextFrameConfig(size_t stream_index,
                                                  uint32_t rtp_timestamp) {
  RTC_DCHECK_LT(stream_index, controllers_.size());
  return controllers_[stream_index]->NextFrameConfig(0, rtp_timestamp);
}

void Vp8TemporalLayers::OnEncodeDone(size_t stream_index,
                                     uint32_t rtp_timestamp,
                                     size_t size_bytes,
                                     bool is_keyframe,
                                     int qp,
                                     CodecSpecificInfo* info) {
  RTC_DCHECK_LT(stream_index, controllers_.size());
  return controllers_[stream_index]->OnEncodeDone(0, rtp_timestamp, size_bytes,
                                                  is_keyframe, qp, info);
}

void Vp8TemporalLayers::OnFrameDropped(size_t stream_index,
                                       uint32_t rtp_timestamp) {
  RTC_DCHECK_LT(stream_index, controllers_.size());
  controllers_[stream_index]->OnFrameDropped(stream_index, rtp_timestamp);
}

void Vp8TemporalLayers::OnPacketLossRateUpdate(float packet_loss_rate) {
  for (auto& controller : controllers_) {
    controller->OnPacketLossRateUpdate(packet_loss_rate);
  }
}

void Vp8TemporalLayers::OnRttUpdate(int64_t rtt_ms) {
  for (auto& controller : controllers_) {
    controller->OnRttUpdate(rtt_ms);
  }
}

void Vp8TemporalLayers::OnLossNotification(
    const VideoEncoder::LossNotification& loss_notification) {
  for (auto& controller : controllers_) {
    controller->OnLossNotification(loss_notification);
  }
}

}  // namespace webrtc
