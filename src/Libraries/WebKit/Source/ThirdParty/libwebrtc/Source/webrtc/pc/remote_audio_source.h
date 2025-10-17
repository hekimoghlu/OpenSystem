/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 2, 2023.
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
#ifndef PC_REMOTE_AUDIO_SOURCE_H_
#define PC_REMOTE_AUDIO_SOURCE_H_

#include <stdint.h>

#include <list>
#include <optional>
#include <string>

#include "api/call/audio_sink.h"
#include "api/media_stream_interface.h"
#include "api/notifier.h"
#include "api/task_queue/task_queue_base.h"
#include "media/base/media_channel.h"
#include "rtc_base/synchronization/mutex.h"

namespace webrtc {

// This class implements the audio source used by the remote audio track.
// This class works by configuring itself as a sink with the underlying media
// engine, then when receiving data will fan out to all added sinks.
class RemoteAudioSource : public Notifier<AudioSourceInterface> {
 public:
  // In Unified Plan, receivers map to m= sections and their tracks and sources
  // survive SSRCs being reconfigured. The life cycle of the remote audio source
  // is associated with the life cycle of the m= section, and thus even if an
  // audio channel is destroyed the RemoteAudioSource should kSurvive.
  //
  // In Plan B however, remote audio sources map 1:1 with an SSRCs and if an
  // audio channel is destroyed, the RemoteAudioSource should kEnd.
  enum class OnAudioChannelGoneAction {
    kSurvive,
    kEnd,
  };

  explicit RemoteAudioSource(
      TaskQueueBase* worker_thread,
      OnAudioChannelGoneAction on_audio_channel_gone_action);

  // Register and unregister remote audio source with the underlying media
  // engine.
  void Start(cricket::VoiceMediaReceiveChannelInterface* media_channel,
             std::optional<uint32_t> ssrc);
  void Stop(cricket::VoiceMediaReceiveChannelInterface* media_channel,
            std::optional<uint32_t> ssrc);
  void SetState(SourceState new_state);

  // MediaSourceInterface implementation.
  MediaSourceInterface::SourceState state() const override;
  bool remote() const override;

  // AudioSourceInterface implementation.
  void SetVolume(double volume) override;
  void RegisterAudioObserver(AudioObserver* observer) override;
  void UnregisterAudioObserver(AudioObserver* observer) override;

  void AddSink(AudioTrackSinkInterface* sink) override;
  void RemoveSink(AudioTrackSinkInterface* sink) override;

 protected:
  ~RemoteAudioSource() override;

 private:
  // These are callbacks from the media engine.
  class AudioDataProxy;

  void OnData(const AudioSinkInterface::Data& audio);
  void OnAudioChannelGone();

  TaskQueueBase* const main_thread_;
  TaskQueueBase* const worker_thread_;
  const OnAudioChannelGoneAction on_audio_channel_gone_action_;
  std::list<AudioObserver*> audio_observers_;
  Mutex sink_lock_;
  std::list<AudioTrackSinkInterface*> sinks_;
  SourceState state_;
};

}  // namespace webrtc

#endif  // PC_REMOTE_AUDIO_SOURCE_H_
