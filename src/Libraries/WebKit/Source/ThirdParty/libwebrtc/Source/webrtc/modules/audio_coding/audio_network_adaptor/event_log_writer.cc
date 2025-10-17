/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 5, 2021.
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
#include "modules/audio_coding/audio_network_adaptor/event_log_writer.h"

#include <math.h>

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <optional>
#include <utility>

#include "api/rtc_event_log/rtc_event.h"
#include "api/rtc_event_log/rtc_event_log.h"
#include "logging/rtc_event_log/events/rtc_event_audio_network_adaptation.h"
#include "rtc_base/checks.h"

namespace webrtc {

EventLogWriter::EventLogWriter(RtcEventLog* event_log,
                               int min_bitrate_change_bps,
                               float min_bitrate_change_fraction,
                               float min_packet_loss_change_fraction)
    : event_log_(event_log),
      min_bitrate_change_bps_(min_bitrate_change_bps),
      min_bitrate_change_fraction_(min_bitrate_change_fraction),
      min_packet_loss_change_fraction_(min_packet_loss_change_fraction) {
  RTC_DCHECK(event_log_);
}

EventLogWriter::~EventLogWriter() = default;

void EventLogWriter::MaybeLogEncoderConfig(
    const AudioEncoderRuntimeConfig& config) {
  if (last_logged_config_.num_channels != config.num_channels)
    return LogEncoderConfig(config);
  if (last_logged_config_.enable_dtx != config.enable_dtx)
    return LogEncoderConfig(config);
  if (last_logged_config_.enable_fec != config.enable_fec)
    return LogEncoderConfig(config);
  if (last_logged_config_.frame_length_ms != config.frame_length_ms)
    return LogEncoderConfig(config);
  if ((!last_logged_config_.bitrate_bps && config.bitrate_bps) ||
      (last_logged_config_.bitrate_bps && config.bitrate_bps &&
       std::abs(*last_logged_config_.bitrate_bps - *config.bitrate_bps) >=
           std::min(static_cast<int>(*last_logged_config_.bitrate_bps *
                                     min_bitrate_change_fraction_),
                    min_bitrate_change_bps_))) {
    return LogEncoderConfig(config);
  }
  if ((!last_logged_config_.uplink_packet_loss_fraction &&
       config.uplink_packet_loss_fraction) ||
      (last_logged_config_.uplink_packet_loss_fraction &&
       config.uplink_packet_loss_fraction &&
       fabs(*last_logged_config_.uplink_packet_loss_fraction -
            *config.uplink_packet_loss_fraction) >=
           min_packet_loss_change_fraction_ *
               *last_logged_config_.uplink_packet_loss_fraction)) {
    return LogEncoderConfig(config);
  }
}

void EventLogWriter::LogEncoderConfig(const AudioEncoderRuntimeConfig& config) {
  auto config_copy = std::make_unique<AudioEncoderRuntimeConfig>(config);
  event_log_->Log(
      std::make_unique<RtcEventAudioNetworkAdaptation>(std::move(config_copy)));
  last_logged_config_ = config;
}

}  // namespace webrtc
