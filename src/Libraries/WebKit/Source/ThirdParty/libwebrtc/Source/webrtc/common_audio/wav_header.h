/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 13, 2023.
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
#ifndef COMMON_AUDIO_WAV_HEADER_H_
#define COMMON_AUDIO_WAV_HEADER_H_

#include <stddef.h>
#include <stdint.h>

#include <algorithm>

#include "rtc_base/checks.h"

namespace webrtc {

// Interface providing header reading functionality.
class WavHeaderReader {
 public:
  // Returns the number of bytes read.
  virtual size_t Read(void* buf, size_t num_bytes) = 0;
  virtual bool SeekForward(uint32_t num_bytes) = 0;
  virtual ~WavHeaderReader() = default;
  virtual int64_t GetPosition() = 0;
};

// Possible WAV formats.
enum class WavFormat {
  kWavFormatPcm = 1,        // PCM, each sample of size bytes_per_sample.
  kWavFormatIeeeFloat = 3,  // IEEE float.
  kWavFormatALaw = 6,       // 8-bit ITU-T G.711 A-law.
  kWavFormatMuLaw = 7,      // 8-bit ITU-T G.711 mu-law.
};

// Header sizes for supported WAV formats.
constexpr size_t kPcmWavHeaderSize = 44;
constexpr size_t kIeeeFloatWavHeaderSize = 58;

// Returns the size of the WAV header for the specified format.
constexpr size_t WavHeaderSize(WavFormat format) {
  if (format == WavFormat::kWavFormatPcm) {
    return kPcmWavHeaderSize;
  }
  RTC_CHECK_EQ(format, WavFormat::kWavFormatIeeeFloat);
  return kIeeeFloatWavHeaderSize;
}

// Returns the maximum size of the supported WAV formats.
constexpr size_t MaxWavHeaderSize() {
  return std::max(WavHeaderSize(WavFormat::kWavFormatPcm),
                  WavHeaderSize(WavFormat::kWavFormatIeeeFloat));
}

// Return true if the given parameters will make a well-formed WAV header.
bool CheckWavParameters(size_t num_channels,
                        int sample_rate,
                        WavFormat format,
                        size_t num_samples);

// Write a kWavHeaderSize bytes long WAV header to buf. The payload that
// follows the header is supposed to have the specified number of interleaved
// channels and contain the specified total number of samples of the specified
// type. The size of the header is returned in header_size. CHECKs the input
// parameters for validity.
void WriteWavHeader(size_t num_channels,
                    int sample_rate,
                    WavFormat format,
                    size_t num_samples,
                    uint8_t* buf,
                    size_t* header_size);

// Read a WAV header from an implemented WavHeaderReader and parse the values
// into the provided output parameters. WavHeaderReader is used because the
// header can be variably sized. Returns false if the header is invalid.
bool ReadWavHeader(WavHeaderReader* readable,
                   size_t* num_channels,
                   int* sample_rate,
                   WavFormat* format,
                   size_t* bytes_per_sample,
                   size_t* num_samples,
                   int64_t* data_start_pos);

}  // namespace webrtc

#endif  // COMMON_AUDIO_WAV_HEADER_H_
