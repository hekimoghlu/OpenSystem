/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 1, 2022.
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
#ifndef API_VOIP_VOIP_ENGINE_FACTORY_H_
#define API_VOIP_VOIP_ENGINE_FACTORY_H_

#include <memory>

#include "api/audio/audio_device.h"
#include "api/audio/audio_processing.h"
#include "api/audio_codecs/audio_decoder_factory.h"
#include "api/audio_codecs/audio_encoder_factory.h"
#include "api/scoped_refptr.h"
#include "api/task_queue/task_queue_factory.h"
#include "api/voip/voip_engine.h"

namespace webrtc {

// VoipEngineConfig is a struct that defines parameters to instantiate a
// VoipEngine instance through CreateVoipEngine factory method. Each member is
// marked with comments as either mandatory or optional and default
// implementations that applications can use.
struct VoipEngineConfig {
  // TODO: bugs.webrtc.org/369904700 - Remove explicit default constructors
  // when deprecated `audio_processing` is removed and thus implicit
  // constructors won't be considered deprecated.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
  VoipEngineConfig() = default;
  VoipEngineConfig(VoipEngineConfig&&) = default;
  VoipEngineConfig& operator=(VoipEngineConfig&&) = default;
#pragma clang diagnostic pop

  // Mandatory (e.g. api/audio_codec/builtin_audio_encoder_factory).
  // AudioEncoderFactory provides a set of audio codecs for VoipEngine to encode
  // the audio input sample. Application can choose to limit the set to reduce
  // application footprint.
  scoped_refptr<AudioEncoderFactory> encoder_factory;

  // Mandatory (e.g. api/audio_codec/builtin_audio_decoder_factory).
  // AudioDecoderFactory provides a set of audio codecs for VoipEngine to decode
  // the received RTP packets from remote media endpoint. Application can choose
  // to limit the set to reduce application footprint.
  scoped_refptr<AudioDecoderFactory> decoder_factory;

  // Mandatory (e.g. api/task_queue/default_task_queue_factory).
  // TaskQeueuFactory provided for VoipEngine to work asynchronously on its
  // encoding flow.
  std::unique_ptr<TaskQueueFactory> task_queue_factory;

  // Mandatory (e.g. modules/audio_device/include).
  // AudioDeviceModule that periocally provides audio input samples from
  // recording device (e.g. microphone) and requests audio output samples to
  // play through its output device (e.g. speaker).
  scoped_refptr<AudioDeviceModule> audio_device_module;

  // Optional (e.g. api/audio/builtin_audio_processing_builder).
  // AudioProcessing provides audio procesing functionalities (e.g. acoustic
  // echo cancellation, noise suppression, gain control, etc) on audio input
  // samples for VoipEngine. When optionally not set, VoipEngine will not have
  // such functionalities to perform on audio input samples received from
  // AudioDeviceModule.
  std::unique_ptr<AudioProcessingBuilderInterface> audio_processing_builder;

  // TODO: bugs.webrtc.org/369904700 - Remove when users are migrated to set
  // `audio_processing_builder` instead.
  [[deprecated]] scoped_refptr<AudioProcessing> audio_processing;
};

// Creates a VoipEngine instance with provided VoipEngineConfig.
std::unique_ptr<VoipEngine> CreateVoipEngine(VoipEngineConfig config);

}  // namespace webrtc

#endif  // API_VOIP_VOIP_ENGINE_FACTORY_H_
