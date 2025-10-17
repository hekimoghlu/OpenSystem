/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 16, 2023.
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
#ifndef COMMON_AUDIO_WAV_FILE_H_
#define COMMON_AUDIO_WAV_FILE_H_

#include <stdint.h>

#include <cstddef>
#include <string>

#include "common_audio/wav_header.h"
#include "rtc_base/system/file_wrapper.h"

namespace webrtc {

// Interface to provide access WAV file parameters.
class WavFile {
 public:
  enum class SampleFormat { kInt16, kFloat };

  virtual ~WavFile() {}

  virtual int sample_rate() const = 0;
  virtual size_t num_channels() const = 0;
  virtual size_t num_samples() const = 0;
};

// Simple C++ class for writing 16-bit integer and 32 bit floating point PCM WAV
// files. All error handling is by calls to RTC_CHECK(), making it unsuitable
// for anything but debug code.
class WavWriter final : public WavFile {
 public:
  // Opens a new WAV file for writing.
  WavWriter(absl::string_view filename,
            int sample_rate,
            size_t num_channels,
            SampleFormat sample_format = SampleFormat::kInt16);
  WavWriter(FileWrapper file,
            int sample_rate,
            size_t num_channels,
            SampleFormat sample_format = SampleFormat::kInt16);

  // Closes the WAV file, after writing its header.
  ~WavWriter() { Close(); }

  WavWriter(const WavWriter&) = delete;
  WavWriter& operator=(const WavWriter&) = delete;

  // Write additional samples to the file. Each sample is in the range
  // [-32768.0,32767.0], and there must be the previously specified number of
  // interleaved channels.
  void WriteSamples(const float* samples, size_t num_samples);
  void WriteSamples(const int16_t* samples, size_t num_samples);

  int sample_rate() const override { return sample_rate_; }
  size_t num_channels() const override { return num_channels_; }
  size_t num_samples() const override { return num_samples_written_; }

 private:
  void Close();
  const int sample_rate_;
  const size_t num_channels_;
  size_t num_samples_written_;
  WavFormat format_;
  FileWrapper file_;
};

// Follows the conventions of WavWriter.
class WavReader final : public WavFile {
 public:
  // Opens an existing WAV file for reading.
  explicit WavReader(absl::string_view filename);
  explicit WavReader(FileWrapper file);

  // Close the WAV file.
  ~WavReader() { Close(); }

  WavReader(const WavReader&) = delete;
  WavReader& operator=(const WavReader&) = delete;

  // Resets position to the beginning of the file.
  void Reset();

  // Returns the number of samples read. If this is less than requested,
  // verifies that the end of the file was reached.
  size_t ReadSamples(size_t num_samples, float* samples);
  size_t ReadSamples(size_t num_samples, int16_t* samples);

  int sample_rate() const override { return sample_rate_; }
  size_t num_channels() const override { return num_channels_; }
  size_t num_samples() const override { return num_samples_in_file_; }

 private:
  void Close();
  int sample_rate_;
  size_t num_channels_;
  WavFormat format_;
  size_t num_samples_in_file_;
  size_t num_unread_samples_;
  FileWrapper file_;
  int64_t
      data_start_pos_;  // Position in the file immediately after WAV header.
};

}  // namespace webrtc

#endif  // COMMON_AUDIO_WAV_FILE_H_
