/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 17, 2023.
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
#ifndef PC_AUDIO_TRACK_H_
#define PC_AUDIO_TRACK_H_

#include <string>

#include "api/media_stream_interface.h"
#include "api/media_stream_track.h"
#include "api/scoped_refptr.h"
#include "api/sequence_checker.h"
#include "rtc_base/system/no_unique_address.h"

namespace webrtc {

// TODO(tommi): Instead of inheriting from `MediaStreamTrack<>`, implement the
// properties directly in this class. `MediaStreamTrack` doesn't guard against
// conflicting access, so we'd need to override those methods anyway in this
// class in order to make sure things are correctly checked.
class AudioTrack : public MediaStreamTrack<AudioTrackInterface>,
                   public ObserverInterface {
 protected:
  // Protected ctor to force use of factory method.
  AudioTrack(absl::string_view label,
             const rtc::scoped_refptr<AudioSourceInterface>& source);

  AudioTrack() = delete;
  AudioTrack(const AudioTrack&) = delete;
  AudioTrack& operator=(const AudioTrack&) = delete;

  ~AudioTrack() override;

 public:
  static rtc::scoped_refptr<AudioTrack> Create(
      absl::string_view id,
      const rtc::scoped_refptr<AudioSourceInterface>& source);

  // MediaStreamTrack implementation.
  std::string kind() const override;

  // AudioTrackInterface implementation.
  AudioSourceInterface* GetSource() const override;

  void AddSink(AudioTrackSinkInterface* sink) override;
  void RemoveSink(AudioTrackSinkInterface* sink) override;

 private:
  // ObserverInterface implementation.
  void OnChanged() override;

 private:
  const rtc::scoped_refptr<AudioSourceInterface> audio_source_;
  RTC_NO_UNIQUE_ADDRESS SequenceChecker signaling_thread_checker_;
};

}  // namespace webrtc

#endif  // PC_AUDIO_TRACK_H_
