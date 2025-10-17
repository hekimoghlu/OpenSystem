/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 30, 2023.
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
#include "pc/audio_track.h"

#include "rtc_base/checks.h"

namespace webrtc {

// static
rtc::scoped_refptr<AudioTrack> AudioTrack::Create(
    absl::string_view id,
    const rtc::scoped_refptr<AudioSourceInterface>& source) {
  return rtc::make_ref_counted<AudioTrack>(id, source);
}

AudioTrack::AudioTrack(absl::string_view label,
                       const rtc::scoped_refptr<AudioSourceInterface>& source)
    : MediaStreamTrack<AudioTrackInterface>(label), audio_source_(source) {
  if (audio_source_) {
    audio_source_->RegisterObserver(this);
    OnChanged();
  }
}

AudioTrack::~AudioTrack() {
  RTC_DCHECK_RUN_ON(&signaling_thread_checker_);
  set_state(MediaStreamTrackInterface::kEnded);
  if (audio_source_)
    audio_source_->UnregisterObserver(this);
}

std::string AudioTrack::kind() const {
  return kAudioKind;
}

AudioSourceInterface* AudioTrack::GetSource() const {
  // Callable from any thread.
  return audio_source_.get();
}

void AudioTrack::AddSink(AudioTrackSinkInterface* sink) {
  RTC_DCHECK_RUN_ON(&signaling_thread_checker_);
  if (audio_source_)
    audio_source_->AddSink(sink);
}

void AudioTrack::RemoveSink(AudioTrackSinkInterface* sink) {
  RTC_DCHECK_RUN_ON(&signaling_thread_checker_);
  if (audio_source_)
    audio_source_->RemoveSink(sink);
}

void AudioTrack::OnChanged() {
  RTC_DCHECK_RUN_ON(&signaling_thread_checker_);
  if (audio_source_->state() == MediaSourceInterface::kEnded) {
    set_state(kEnded);
  } else {
    set_state(kLive);
  }
}

}  // namespace webrtc
