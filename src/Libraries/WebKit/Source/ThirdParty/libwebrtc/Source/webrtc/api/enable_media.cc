/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 9, 2024.
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
#include "api/enable_media.h"

#include <memory>
#include <utility>

#include "absl/base/nullability.h"
#include "api/environment/environment.h"
#include "api/peer_connection_interface.h"
#include "api/scoped_refptr.h"
#include "call/call.h"
#include "call/call_config.h"
#include "media/base/media_engine.h"
#include "media/engine/webrtc_video_engine.h"
#include "media/engine/webrtc_voice_engine.h"
#include "pc/media_factory.h"

namespace webrtc {
namespace {

using ::cricket::CompositeMediaEngine;
using ::cricket::MediaEngineInterface;
using ::cricket::WebRtcVideoEngine;
using ::cricket::WebRtcVoiceEngine;

class MediaFactoryImpl : public MediaFactory {
 public:
  MediaFactoryImpl() = default;
  MediaFactoryImpl(const MediaFactoryImpl&) = delete;
  MediaFactoryImpl& operator=(const MediaFactoryImpl&) = delete;
  ~MediaFactoryImpl() override = default;

  std::unique_ptr<Call> CreateCall(CallConfig config) override {
    return webrtc::Call::Create(std::move(config));
  }

  std::unique_ptr<MediaEngineInterface> CreateMediaEngine(
      const Environment& env,
      PeerConnectionFactoryDependencies& deps) override {
    absl::Nullable<scoped_refptr<AudioProcessing>> audio_processing =
        deps.audio_processing_builder != nullptr
            ? std::move(deps.audio_processing_builder)->Build(env)
            : std::move(deps.audio_processing);

    auto audio_engine = std::make_unique<WebRtcVoiceEngine>(
        &env.task_queue_factory(), deps.adm.get(),
        std::move(deps.audio_encoder_factory),
        std::move(deps.audio_decoder_factory), std::move(deps.audio_mixer),
        std::move(audio_processing), std::move(deps.audio_frame_processor),
        env.field_trials());
    auto video_engine = std::make_unique<WebRtcVideoEngine>(
        std::move(deps.video_encoder_factory),
        std::move(deps.video_decoder_factory), env.field_trials());
    return std::make_unique<CompositeMediaEngine>(std::move(audio_engine),
                                                  std::move(video_engine));
  }
};

}  // namespace

void EnableMedia(PeerConnectionFactoryDependencies& deps) {
  if (deps.media_factory != nullptr) {
    // Do nothing if media is already enabled. Overwriting media_factory can be
    // harmful when a different (e.g. test-only) implementation is used.
    return;
  }
  deps.media_factory = std::make_unique<MediaFactoryImpl>();
}

}  // namespace webrtc
