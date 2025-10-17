/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 10, 2022.
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
#ifndef MODULES_AUDIO_PROCESSING_AGC_AGC_H_
#define MODULES_AUDIO_PROCESSING_AGC_AGC_H_

#include <memory>

#include "api/array_view.h"
#include "modules/audio_processing/vad/voice_activity_detector.h"

namespace webrtc {

class LoudnessHistogram;

class Agc {
 public:
  Agc();
  virtual ~Agc();

  // `audio` must be mono; in a multi-channel stream, provide the first (usually
  // left) channel.
  virtual void Process(rtc::ArrayView<const int16_t> audio);

  // Retrieves the difference between the target RMS level and the current
  // signal RMS level in dB. Returns true if an update is available and false
  // otherwise, in which case `error` should be ignored and no action taken.
  virtual bool GetRmsErrorDb(int* error);
  virtual void Reset();

  virtual int set_target_level_dbfs(int level);
  virtual int target_level_dbfs() const;
  virtual float voice_probability() const;

 private:
  double target_level_loudness_;
  int target_level_dbfs_;
  std::unique_ptr<LoudnessHistogram> histogram_;
  std::unique_ptr<LoudnessHistogram> inactive_histogram_;
  VoiceActivityDetector vad_;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_AGC_AGC_H_
