/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 17, 2023.
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
#ifndef MODULES_AUDIO_CODING_AUDIO_NETWORK_ADAPTOR_MOCK_MOCK_AUDIO_NETWORK_ADAPTOR_H_
#define MODULES_AUDIO_CODING_AUDIO_NETWORK_ADAPTOR_MOCK_MOCK_AUDIO_NETWORK_ADAPTOR_H_

#include "modules/audio_coding/audio_network_adaptor/include/audio_network_adaptor.h"
#include "test/gmock.h"

namespace webrtc {

class MockAudioNetworkAdaptor : public AudioNetworkAdaptor {
 public:
  ~MockAudioNetworkAdaptor() override { Die(); }
  MOCK_METHOD(void, Die, ());

  MOCK_METHOD(void, SetUplinkBandwidth, (int uplink_bandwidth_bps), (override));

  MOCK_METHOD(void,
              SetUplinkPacketLossFraction,
              (float uplink_packet_loss_fraction),
              (override));

  MOCK_METHOD(void, SetRtt, (int rtt_ms), (override));

  MOCK_METHOD(void,
              SetTargetAudioBitrate,
              (int target_audio_bitrate_bps),
              (override));

  MOCK_METHOD(void,
              SetOverhead,
              (size_t overhead_bytes_per_packet),
              (override));

  MOCK_METHOD(AudioEncoderRuntimeConfig,
              GetEncoderRuntimeConfig,
              (),
              (override));

  MOCK_METHOD(void, StartDebugDump, (FILE * file_handle), (override));

  MOCK_METHOD(void, StopDebugDump, (), (override));

  MOCK_METHOD(ANAStats, GetStats, (), (const, override));
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_CODING_AUDIO_NETWORK_ADAPTOR_MOCK_MOCK_AUDIO_NETWORK_ADAPTOR_H_
