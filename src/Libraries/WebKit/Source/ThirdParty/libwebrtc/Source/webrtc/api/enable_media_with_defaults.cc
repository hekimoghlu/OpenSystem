/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 9, 2021.
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
#include "api/enable_media_with_defaults.h"

#include <memory>

#include "api/audio/builtin_audio_processing_builder.h"
#include "api/audio_codecs/builtin_audio_decoder_factory.h"
#include "api/audio_codecs/builtin_audio_encoder_factory.h"
#include "api/enable_media.h"
#include "api/peer_connection_interface.h"
#include "api/scoped_refptr.h"
#include "api/task_queue/default_task_queue_factory.h"
#include "api/video_codecs/builtin_video_decoder_factory.h"
#include "api/video_codecs/builtin_video_encoder_factory.h"

namespace webrtc {

void EnableMediaWithDefaults(PeerConnectionFactoryDependencies& deps) {
  if (deps.task_queue_factory == nullptr) {
    deps.task_queue_factory = CreateDefaultTaskQueueFactory();
  }
  if (deps.audio_encoder_factory == nullptr) {
    deps.audio_encoder_factory = CreateBuiltinAudioEncoderFactory();
  }
  if (deps.audio_decoder_factory == nullptr) {
    deps.audio_decoder_factory = CreateBuiltinAudioDecoderFactory();
  }
  if (deps.audio_processing == nullptr &&
      deps.audio_processing_builder == nullptr) {
    deps.audio_processing_builder =
        std::make_unique<BuiltinAudioProcessingBuilder>();
  }
  if (deps.video_encoder_factory == nullptr) {
    deps.video_encoder_factory = CreateBuiltinVideoEncoderFactory();
  }
  if (deps.video_decoder_factory == nullptr) {
    deps.video_decoder_factory = CreateBuiltinVideoDecoderFactory();
  }
  EnableMedia(deps);
}

}  // namespace webrtc
