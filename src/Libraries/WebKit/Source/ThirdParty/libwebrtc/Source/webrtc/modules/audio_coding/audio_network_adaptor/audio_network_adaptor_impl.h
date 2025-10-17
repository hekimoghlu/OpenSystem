/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 1, 2022.
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
#ifndef MODULES_AUDIO_CODING_AUDIO_NETWORK_ADAPTOR_AUDIO_NETWORK_ADAPTOR_IMPL_H_
#define MODULES_AUDIO_CODING_AUDIO_NETWORK_ADAPTOR_AUDIO_NETWORK_ADAPTOR_IMPL_H_

#include <stdio.h>

#include <memory>
#include <optional>

#include "api/audio_codecs/audio_encoder.h"
#include "modules/audio_coding/audio_network_adaptor/controller.h"
#include "modules/audio_coding/audio_network_adaptor/debug_dump_writer.h"
#include "modules/audio_coding/audio_network_adaptor/include/audio_network_adaptor.h"
#include "modules/audio_coding/audio_network_adaptor/include/audio_network_adaptor_config.h"

namespace webrtc {

class ControllerManager;
class EventLogWriter;
class RtcEventLog;

class AudioNetworkAdaptorImpl final : public AudioNetworkAdaptor {
 public:
  struct Config {
    Config();
    ~Config();
    RtcEventLog* event_log;
  };

  AudioNetworkAdaptorImpl(
      const Config& config,
      std::unique_ptr<ControllerManager> controller_manager,
      std::unique_ptr<DebugDumpWriter> debug_dump_writer = nullptr);

  ~AudioNetworkAdaptorImpl() override;

  AudioNetworkAdaptorImpl(const AudioNetworkAdaptorImpl&) = delete;
  AudioNetworkAdaptorImpl& operator=(const AudioNetworkAdaptorImpl&) = delete;

  void SetUplinkBandwidth(int uplink_bandwidth_bps) override;

  void SetUplinkPacketLossFraction(float uplink_packet_loss_fraction) override;

  void SetRtt(int rtt_ms) override;

  void SetTargetAudioBitrate(int target_audio_bitrate_bps) override;

  void SetOverhead(size_t overhead_bytes_per_packet) override;

  AudioEncoderRuntimeConfig GetEncoderRuntimeConfig() override;

  void StartDebugDump(FILE* file_handle) override;

  void StopDebugDump() override;

  ANAStats GetStats() const override;

 private:
  void DumpNetworkMetrics();

  void UpdateNetworkMetrics(const Controller::NetworkMetrics& network_metrics);

  const Config config_;

  std::unique_ptr<ControllerManager> controller_manager_;

  std::unique_ptr<DebugDumpWriter> debug_dump_writer_;

  const std::unique_ptr<EventLogWriter> event_log_writer_;

  Controller::NetworkMetrics last_metrics_;

  std::optional<AudioEncoderRuntimeConfig> prev_config_;

  ANAStats stats_;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_CODING_AUDIO_NETWORK_ADAPTOR_AUDIO_NETWORK_ADAPTOR_IMPL_H_
