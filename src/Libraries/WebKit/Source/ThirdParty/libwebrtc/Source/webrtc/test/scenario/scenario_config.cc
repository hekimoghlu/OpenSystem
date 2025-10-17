/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 7, 2025.
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
#include "test/scenario/scenario_config.h"

namespace webrtc {
namespace test {

TransportControllerConfig::Rates::Rates() = default;
TransportControllerConfig::Rates::Rates(
    const TransportControllerConfig::Rates&) = default;
TransportControllerConfig::Rates::~Rates() = default;

PacketStreamConfig::PacketStreamConfig() = default;
PacketStreamConfig::PacketStreamConfig(const PacketStreamConfig&) = default;
PacketStreamConfig::~PacketStreamConfig() = default;

VideoStreamConfig::Encoder::Encoder() = default;
VideoStreamConfig::Encoder::Encoder(const VideoStreamConfig::Encoder&) =
    default;
VideoStreamConfig::Encoder::~Encoder() = default;

VideoStreamConfig::Stream::Stream() = default;
VideoStreamConfig::Stream::Stream(const VideoStreamConfig::Stream&) = default;
VideoStreamConfig::Stream::~Stream() = default;

AudioStreamConfig::AudioStreamConfig() = default;
AudioStreamConfig::AudioStreamConfig(const AudioStreamConfig&) = default;
AudioStreamConfig::~AudioStreamConfig() = default;

AudioStreamConfig::Encoder::Encoder() = default;
AudioStreamConfig::Encoder::Encoder(const AudioStreamConfig::Encoder&) =
    default;
AudioStreamConfig::Encoder::~Encoder() = default;

AudioStreamConfig::Stream::Stream() = default;
AudioStreamConfig::Stream::Stream(const AudioStreamConfig::Stream&) = default;
AudioStreamConfig::Stream::~Stream() = default;

}  // namespace test
}  // namespace webrtc
