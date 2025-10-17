/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 14, 2023.
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
#include "api/create_peerconnection_factory.h"

#include <memory>
#include <utility>

#include "api/audio/audio_device.h"
#include "api/audio/audio_mixer.h"
#include "api/audio/audio_processing.h"
#include "api/audio/builtin_audio_processing_builder.h"
#include "api/audio_codecs/audio_decoder_factory.h"
#include "api/audio_codecs/audio_encoder_factory.h"
#include "api/enable_media.h"
#include "api/field_trials_view.h"
#include "api/peer_connection_interface.h"
#include "api/rtc_event_log/rtc_event_log_factory.h"
#if defined(WEBRTC_WEBKIT_BUILD)
#include "api/task_queue/default_task_queue_factory.h"
#endif
#include "api/scoped_refptr.h"
#include "api/video_codecs/video_decoder_factory.h"
#include "api/video_codecs/video_encoder_factory.h"
#include "rtc_base/thread.h"

namespace webrtc {

rtc::scoped_refptr<PeerConnectionFactoryInterface> CreatePeerConnectionFactory(
    rtc::Thread* network_thread,
    rtc::Thread* worker_thread,
    rtc::Thread* signaling_thread,
    rtc::scoped_refptr<AudioDeviceModule> default_adm,
    rtc::scoped_refptr<AudioEncoderFactory> audio_encoder_factory,
    rtc::scoped_refptr<AudioDecoderFactory> audio_decoder_factory,
    std::unique_ptr<VideoEncoderFactory> video_encoder_factory,
    std::unique_ptr<VideoDecoderFactory> video_decoder_factory,
    rtc::scoped_refptr<AudioMixer> audio_mixer,
    rtc::scoped_refptr<AudioProcessing> audio_processing,
    std::unique_ptr<AudioFrameProcessor> audio_frame_processor,
    std::unique_ptr<FieldTrialsView> field_trials
#if defined(WEBRTC_WEBKIT_BUILD)
    , std::unique_ptr<TaskQueueFactory> task_queue_factory
#endif
  ) {
  if (!field_trials) {
    field_trials = std::make_unique<webrtc::FieldTrialBasedConfig>();
  }

  PeerConnectionFactoryDependencies dependencies;
  dependencies.network_thread = network_thread;
  dependencies.worker_thread = worker_thread;
  dependencies.signaling_thread = signaling_thread;
#if defined(WEBRTC_WEBKIT_BUILD)
  dependencies.task_queue_factory = task_queue_factory ? std::move(task_queue_factory) : CreateDefaultTaskQueueFactory(field_trials.get());
#else
  dependencies.task_queue_factory =
      CreateDefaultTaskQueueFactory(field_trials.get());
#endif
  dependencies.event_log_factory = std::make_unique<RtcEventLogFactory>();
  dependencies.trials = std::move(field_trials);

  if (network_thread) {
    // TODO(bugs.webrtc.org/13145): Add an rtc::SocketFactory* argument.
    dependencies.socket_factory = network_thread->socketserver();
  }
  dependencies.adm = std::move(default_adm);
  dependencies.audio_encoder_factory = std::move(audio_encoder_factory);
  dependencies.audio_decoder_factory = std::move(audio_decoder_factory);
  dependencies.audio_frame_processor = std::move(audio_frame_processor);
  if (audio_processing != nullptr) {
    dependencies.audio_processing_builder =
        CustomAudioProcessing(std::move(audio_processing));
  } else {
#ifndef WEBRTC_EXCLUDE_AUDIO_PROCESSING_MODULE
    dependencies.audio_processing_builder =
        std::make_unique<BuiltinAudioProcessingBuilder>();
#endif
  }
  dependencies.audio_mixer = std::move(audio_mixer);
  dependencies.video_encoder_factory = std::move(video_encoder_factory);
  dependencies.video_decoder_factory = std::move(video_decoder_factory);
  EnableMedia(dependencies);

  return CreateModularPeerConnectionFactory(std::move(dependencies));
}

}  // namespace webrtc
