/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 23, 2022.
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
#ifndef MODULES_AUDIO_CODING_AUDIO_NETWORK_ADAPTOR_EVENT_LOG_WRITER_H_
#define MODULES_AUDIO_CODING_AUDIO_NETWORK_ADAPTOR_EVENT_LOG_WRITER_H_

#include "modules/audio_coding/audio_network_adaptor/include/audio_network_adaptor_config.h"

namespace webrtc {
class RtcEventLog;

class EventLogWriter final {
 public:
  EventLogWriter(RtcEventLog* event_log,
                 int min_bitrate_change_bps,
                 float min_bitrate_change_fraction,
                 float min_packet_loss_change_fraction);
  ~EventLogWriter();

  EventLogWriter(const EventLogWriter&) = delete;
  EventLogWriter& operator=(const EventLogWriter&) = delete;

  void MaybeLogEncoderConfig(const AudioEncoderRuntimeConfig& config);

 private:
  void LogEncoderConfig(const AudioEncoderRuntimeConfig& config);

  RtcEventLog* const event_log_;
  const int min_bitrate_change_bps_;
  const float min_bitrate_change_fraction_;
  const float min_packet_loss_change_fraction_;
  AudioEncoderRuntimeConfig last_logged_config_;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_CODING_AUDIO_NETWORK_ADAPTOR_EVENT_LOG_WRITER_H_
