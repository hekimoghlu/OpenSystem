/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 24, 2024.
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
#ifndef API_VOIP_VOIP_ENGINE_H_
#define API_VOIP_VOIP_ENGINE_H_

namespace webrtc {

class VoipBase;
class VoipCodec;
class VoipNetwork;
class VoipDtmf;
class VoipStatistics;
class VoipVolumeControl;

// VoipEngine is the main interface serving as the entry point for all VoIP
// APIs. A single instance of VoipEngine should suffice the most of the need for
// typical VoIP applications as it handles multiple media sessions including a
// specialized session type like ad-hoc conference. Below example code
// describes the typical sequence of API usage. Each API header contains more
// description on what the methods are used for.
//
//   // Caller is responsible of setting desired audio components.
//   VoipEngineConfig config;
//   config.encoder_factory = CreateBuiltinAudioEncoderFactory();
//   config.decoder_factory = CreateBuiltinAudioDecoderFactory();
//   config.task_queue_factory = CreateDefaultTaskQueueFactory();
//   config.audio_device =
//       AudioDeviceModule::Create(AudioDeviceModule::kPlatformDefaultAudio,
//                                 config.task_queue_factory.get());
//   config.audio_processing_builder =
//       std::make_unique<BuiltinAudioProcessingBuilder>();
//
//   auto voip_engine = CreateVoipEngine(std::move(config));
//
//   auto& voip_base = voip_engine->Base();
//   auto& voip_codec = voip_engine->Codec();
//   auto& voip_network = voip_engine->Network();
//
//   ChannelId channel = voip_base.CreateChannel(&app_transport_);
//
//   // After SDP offer/answer, set payload type and codecs that have been
//   // decided through SDP negotiation.
//   // VoipResult handling omitted here.
//   voip_codec.SetSendCodec(channel, ...);
//   voip_codec.SetReceiveCodecs(channel, ...);
//
//   // Start sending and playing RTP on voip channel.
//   // VoipResult handling omitted here.
//   voip_base.StartSend(channel);
//   voip_base.StartPlayout(channel);
//
//   // Inject received RTP/RTCP through VoipNetwork interface.
//   // VoipResult handling omitted here.
//   voip_network.ReceivedRTPPacket(channel, ...);
//   voip_network.ReceivedRTCPPacket(channel, ...);
//
//   // Stop and release voip channel.
//   // VoipResult handling omitted here.
//   voip_base.StopSend(channel);
//   voip_base.StopPlayout(channel);
//   voip_base.ReleaseChannel(channel);
//
class VoipEngine {
 public:
  virtual ~VoipEngine() = default;

  // VoipBase is the audio session management interface that
  // creates/releases/starts/stops an one-to-one audio media session.
  virtual VoipBase& Base() = 0;

  // VoipNetwork provides injection APIs that would enable application
  // to send and receive RTP/RTCP packets. There is no default network module
  // that provides RTP transmission and reception.
  virtual VoipNetwork& Network() = 0;

  // VoipCodec provides codec configuration APIs for encoder and decoders.
  virtual VoipCodec& Codec() = 0;

  // VoipDtmf provides DTMF event APIs to register and send DTMF events.
  virtual VoipDtmf& Dtmf() = 0;

  // VoipStatistics provides performance metrics around audio decoding module
  // and jitter buffer (NetEq).
  virtual VoipStatistics& Statistics() = 0;

  // VoipVolumeControl provides various input/output volume control.
  virtual VoipVolumeControl& VolumeControl() = 0;
};

}  // namespace webrtc

#endif  // API_VOIP_VOIP_ENGINE_H_
