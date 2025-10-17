/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 7, 2025.
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
#ifndef MODULES_AUDIO_MIXER_AUDIO_MIXER_IMPL_H_
#define MODULES_AUDIO_MIXER_AUDIO_MIXER_IMPL_H_

#include <stddef.h>

#include <memory>
#include <vector>

#include "api/array_view.h"
#include "api/audio/audio_frame.h"
#include "api/audio/audio_mixer.h"
#include "api/scoped_refptr.h"
#include "modules/audio_mixer/frame_combiner.h"
#include "modules/audio_mixer/output_rate_calculator.h"
#include "rtc_base/race_checker.h"
#include "rtc_base/synchronization/mutex.h"
#include "rtc_base/thread_annotations.h"

namespace webrtc {

class AudioMixerImpl : public AudioMixer {
 public:
  struct SourceStatus;

  // AudioProcessing only accepts 10 ms frames.
  static const int kFrameDurationInMs = 10;

  static rtc::scoped_refptr<AudioMixerImpl> Create();

  static rtc::scoped_refptr<AudioMixerImpl> Create(
      std::unique_ptr<OutputRateCalculator> output_rate_calculator,
      bool use_limiter);

  ~AudioMixerImpl() override;

  AudioMixerImpl(const AudioMixerImpl&) = delete;
  AudioMixerImpl& operator=(const AudioMixerImpl&) = delete;

  // AudioMixer functions
  bool AddSource(Source* audio_source) override;
  void RemoveSource(Source* audio_source) override;

  void Mix(size_t number_of_channels,
           AudioFrame* audio_frame_for_mixing) override
      RTC_LOCKS_EXCLUDED(mutex_);

 protected:
  AudioMixerImpl(std::unique_ptr<OutputRateCalculator> output_rate_calculator,
                 bool use_limiter);

 private:
  struct HelperContainers;

  void UpdateSourceCountStats() RTC_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // Fetches audio frames to mix from sources.
  rtc::ArrayView<AudioFrame* const> GetAudioFromSources(int output_frequency)
      RTC_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // The critical section lock guards audio source insertion and
  // removal, which can be done from any thread. The race checker
  // checks that mixing is done sequentially.
  mutable Mutex mutex_;

  std::unique_ptr<OutputRateCalculator> output_rate_calculator_;

  // List of all audio sources.
  std::vector<std::unique_ptr<SourceStatus>> audio_source_list_
      RTC_GUARDED_BY(mutex_);
  const std::unique_ptr<HelperContainers> helper_containers_
      RTC_GUARDED_BY(mutex_);

  // Component that handles actual adding of audio frames.
  FrameCombiner frame_combiner_;

  // The highest source count this mixer has ever had. Used for UMA stats.
  size_t max_source_count_ever_ = 0;
};
}  // namespace webrtc

#endif  // MODULES_AUDIO_MIXER_AUDIO_MIXER_IMPL_H_
