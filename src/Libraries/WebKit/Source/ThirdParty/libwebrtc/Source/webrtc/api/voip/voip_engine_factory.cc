/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 25, 2022.
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
#include "api/voip/voip_engine_factory.h"

#include <memory>
#include <utility>

#include "api/audio/audio_processing.h"
#include "api/environment/environment.h"
#include "api/environment/environment_factory.h"
#include "api/scoped_refptr.h"
#include "api/voip/voip_engine.h"
#include "audio/voip/voip_core.h"
#include "rtc_base/checks.h"
#include "rtc_base/logging.h"

namespace webrtc {

std::unique_ptr<VoipEngine> CreateVoipEngine(VoipEngineConfig config) {
  RTC_CHECK(config.encoder_factory);
  RTC_CHECK(config.decoder_factory);
  RTC_CHECK(config.task_queue_factory);
  RTC_CHECK(config.audio_device_module);

  Environment env = CreateEnvironment(std::move(config.task_queue_factory));

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
  RTC_CHECK(config.audio_processing == nullptr ||
            config.audio_processing_builder == nullptr);
  scoped_refptr<AudioProcessing> audio_processing =
      std::move(config.audio_processing);
#pragma clang diagnostic pop
  if (config.audio_processing_builder != nullptr) {
    audio_processing = std::move(config.audio_processing_builder)->Build(env);
  }

  if (audio_processing == nullptr) {
    RTC_DLOG(LS_INFO) << "No audio processing functionality provided.";
  }

  return std::make_unique<VoipCore>(
      env, std::move(config.encoder_factory), std::move(config.decoder_factory),
      std::move(config.audio_device_module), std::move(audio_processing));
}

}  // namespace webrtc
