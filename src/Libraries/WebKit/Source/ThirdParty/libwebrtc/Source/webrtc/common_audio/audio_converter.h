/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 20, 2024.
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
#ifndef COMMON_AUDIO_AUDIO_CONVERTER_H_
#define COMMON_AUDIO_AUDIO_CONVERTER_H_

#include <stddef.h>

#include <memory>

namespace webrtc {

// Format conversion (remixing and resampling) for audio. Only simple remixing
// conversions are supported: downmix to mono (i.e. `dst_channels` == 1) or
// upmix from mono (i.e. |src_channels == 1|).
//
// The source and destination chunks have the same duration in time; specifying
// the number of frames is equivalent to specifying the sample rates.
class AudioConverter {
 public:
  // Returns a new AudioConverter, which will use the supplied format for its
  // lifetime. Caller is responsible for the memory.
  static std::unique_ptr<AudioConverter> Create(size_t src_channels,
                                                size_t src_frames,
                                                size_t dst_channels,
                                                size_t dst_frames);
  virtual ~AudioConverter() {}

  AudioConverter(const AudioConverter&) = delete;
  AudioConverter& operator=(const AudioConverter&) = delete;

  // Convert `src`, containing `src_size` samples, to `dst`, having a sample
  // capacity of `dst_capacity`. Both point to a series of buffers containing
  // the samples for each channel. The sizes must correspond to the format
  // passed to Create().
  virtual void Convert(const float* const* src,
                       size_t src_size,
                       float* const* dst,
                       size_t dst_capacity) = 0;

  size_t src_channels() const { return src_channels_; }
  size_t src_frames() const { return src_frames_; }
  size_t dst_channels() const { return dst_channels_; }
  size_t dst_frames() const { return dst_frames_; }

 protected:
  AudioConverter();
  AudioConverter(size_t src_channels,
                 size_t src_frames,
                 size_t dst_channels,
                 size_t dst_frames);

  // Helper to RTC_CHECK that inputs are correctly sized.
  void CheckSizes(size_t src_size, size_t dst_capacity) const;

 private:
  const size_t src_channels_;
  const size_t src_frames_;
  const size_t dst_channels_;
  const size_t dst_frames_;
};

}  // namespace webrtc

#endif  // COMMON_AUDIO_AUDIO_CONVERTER_H_
