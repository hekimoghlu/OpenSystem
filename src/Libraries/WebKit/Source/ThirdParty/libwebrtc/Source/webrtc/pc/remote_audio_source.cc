/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 16, 2024.
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
#include "pc/remote_audio_source.h"

#include <stddef.h>

#include <memory>
#include <string>
#include <utility>

#include "absl/algorithm/container.h"
#include "api/scoped_refptr.h"
#include "api/sequence_checker.h"
#include "api/task_queue/task_queue_base.h"
#include "rtc_base/checks.h"
#include "rtc_base/logging.h"
#include "rtc_base/strings/string_format.h"
#include "rtc_base/trace_event.h"

namespace webrtc {

// This proxy is passed to the underlying media engine to receive audio data as
// they come in. The data will then be passed back up to the RemoteAudioSource
// which will fan it out to all the sinks that have been added to it.
class RemoteAudioSource::AudioDataProxy : public AudioSinkInterface {
 public:
  explicit AudioDataProxy(RemoteAudioSource* source) : source_(source) {
    RTC_DCHECK(source);
  }

  AudioDataProxy() = delete;
  AudioDataProxy(const AudioDataProxy&) = delete;
  AudioDataProxy& operator=(const AudioDataProxy&) = delete;

  ~AudioDataProxy() override { source_->OnAudioChannelGone(); }

  // AudioSinkInterface implementation.
  void OnData(const AudioSinkInterface::Data& audio) override {
    source_->OnData(audio);
  }

 private:
  const rtc::scoped_refptr<RemoteAudioSource> source_;
};

RemoteAudioSource::RemoteAudioSource(
    TaskQueueBase* worker_thread,
    OnAudioChannelGoneAction on_audio_channel_gone_action)
    : main_thread_(TaskQueueBase::Current()),
      worker_thread_(worker_thread),
      on_audio_channel_gone_action_(on_audio_channel_gone_action),
      state_(MediaSourceInterface::kInitializing) {
  RTC_DCHECK(main_thread_);
  RTC_DCHECK(worker_thread_);
}

RemoteAudioSource::~RemoteAudioSource() {
  RTC_DCHECK(audio_observers_.empty());
  if (!sinks_.empty()) {
    RTC_LOG(LS_WARNING)
        << "RemoteAudioSource destroyed while sinks_ is non-empty.";
  }
}

void RemoteAudioSource::Start(
    cricket::VoiceMediaReceiveChannelInterface* media_channel,
    std::optional<uint32_t> ssrc) {
  RTC_DCHECK_RUN_ON(worker_thread_);

  // Register for callbacks immediately before AddSink so that we always get
  // notified when a channel goes out of scope (signaled when "AudioDataProxy"
  // is destroyed).
  RTC_DCHECK(media_channel);
  ssrc ? media_channel->SetRawAudioSink(*ssrc,
                                        std::make_unique<AudioDataProxy>(this))
       : media_channel->SetDefaultRawAudioSink(
             std::make_unique<AudioDataProxy>(this));
}

void RemoteAudioSource::Stop(
    cricket::VoiceMediaReceiveChannelInterface* media_channel,
    std::optional<uint32_t> ssrc) {
  RTC_DCHECK_RUN_ON(worker_thread_);
  RTC_DCHECK(media_channel);
  ssrc ? media_channel->SetRawAudioSink(*ssrc, nullptr)
       : media_channel->SetDefaultRawAudioSink(nullptr);
}

void RemoteAudioSource::SetState(SourceState new_state) {
  RTC_DCHECK_RUN_ON(main_thread_);
  if (state_ != new_state) {
    state_ = new_state;
    FireOnChanged();
  }
}

MediaSourceInterface::SourceState RemoteAudioSource::state() const {
  RTC_DCHECK_RUN_ON(main_thread_);
  return state_;
}

bool RemoteAudioSource::remote() const {
  RTC_DCHECK_RUN_ON(main_thread_);
  return true;
}

void RemoteAudioSource::SetVolume(double volume) {
  RTC_DCHECK_GE(volume, 0);
  RTC_DCHECK_LE(volume, 10);
  RTC_LOG(LS_INFO) << rtc::StringFormat("RAS::%s({volume=%.2f})", __func__,
                                        volume);
  for (auto* observer : audio_observers_) {
    observer->OnSetVolume(volume);
  }
}

void RemoteAudioSource::RegisterAudioObserver(AudioObserver* observer) {
  RTC_DCHECK(observer != NULL);
  RTC_DCHECK(!absl::c_linear_search(audio_observers_, observer));
  audio_observers_.push_back(observer);
}

void RemoteAudioSource::UnregisterAudioObserver(AudioObserver* observer) {
  RTC_DCHECK(observer != NULL);
  audio_observers_.remove(observer);
}

void RemoteAudioSource::AddSink(AudioTrackSinkInterface* sink) {
  RTC_DCHECK_RUN_ON(main_thread_);
  RTC_DCHECK(sink);

  MutexLock lock(&sink_lock_);
  RTC_DCHECK(!absl::c_linear_search(sinks_, sink));
  sinks_.push_back(sink);
}

void RemoteAudioSource::RemoveSink(AudioTrackSinkInterface* sink) {
  RTC_DCHECK_RUN_ON(main_thread_);
  RTC_DCHECK(sink);

  MutexLock lock(&sink_lock_);
  sinks_.remove(sink);
}

void RemoteAudioSource::OnData(const AudioSinkInterface::Data& audio) {
  // Called on the externally-owned audio callback thread, via/from webrtc.
  TRACE_EVENT0("webrtc", "RemoteAudioSource::OnData");
  MutexLock lock(&sink_lock_);
  for (auto* sink : sinks_) {
    // When peerconnection acts as an audio source, it should not provide
    // absolute capture timestamp.
    sink->OnData(audio.data, 16, audio.sample_rate, audio.channels,
                 audio.samples_per_channel,
                 /*absolute_capture_timestamp_ms=*/std::nullopt);
  }
}

void RemoteAudioSource::OnAudioChannelGone() {
  if (on_audio_channel_gone_action_ != OnAudioChannelGoneAction::kEnd) {
    return;
  }
  // Called when the audio channel is deleted. It may be the worker thread or
  // may be a different task queue.
  // This object needs to live long enough for the cleanup logic in the posted
  // task to run, so take a reference to it. Sometimes the task may not be
  // processed (because the task queue was destroyed shortly after this call),
  // but that is fine because the task queue destructor will take care of
  // destroying task which will release the reference on RemoteAudioSource.
  rtc::scoped_refptr<RemoteAudioSource> thiz(this);
  main_thread_->PostTask([thiz = std::move(thiz)] {
    thiz->sinks_.clear();
    thiz->SetState(MediaSourceInterface::kEnded);
  });
}

}  // namespace webrtc
