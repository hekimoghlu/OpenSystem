/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 26, 2025.
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
#ifndef MODULES_AUDIO_CODING_NETEQ_COMFORT_NOISE_H_
#define MODULES_AUDIO_CODING_NETEQ_COMFORT_NOISE_H_

#include <stddef.h>

namespace webrtc {

// Forward declarations.
class AudioMultiVector;
class DecoderDatabase;
class SyncBuffer;
struct Packet;

// This class acts as an interface to the CNG generator.
class ComfortNoise {
 public:
  enum ReturnCodes {
    kOK = 0,
    kUnknownPayloadType,
    kInternalError,
    kMultiChannelNotSupported
  };

  ComfortNoise(int fs_hz,
               DecoderDatabase* decoder_database,
               SyncBuffer* sync_buffer)
      : fs_hz_(fs_hz),
        first_call_(true),
        overlap_length_(5 * fs_hz_ / 8000),
        decoder_database_(decoder_database),
        sync_buffer_(sync_buffer) {}

  ComfortNoise(const ComfortNoise&) = delete;
  ComfortNoise& operator=(const ComfortNoise&) = delete;

  // Resets the state. Should be called before each new comfort noise period.
  void Reset();

  // Update the comfort noise generator with the parameters in `packet`.
  int UpdateParameters(const Packet& packet);

  // Generates `requested_length` samples of comfort noise and writes to
  // `output`. If this is the first in call after Reset (or first after creating
  // the object), it will also mix in comfort noise at the end of the
  // SyncBuffer object provided in the constructor.
  int Generate(size_t requested_length, AudioMultiVector* output);

  // Returns the last error code that was produced by the comfort noise
  // decoder. Returns 0 if no error has been encountered since the last reset.
  int internal_error_code() { return internal_error_code_; }

 private:
  int fs_hz_;
  bool first_call_;
  size_t overlap_length_;
  DecoderDatabase* decoder_database_;
  SyncBuffer* sync_buffer_;
  int internal_error_code_;
};

}  // namespace webrtc
#endif  // MODULES_AUDIO_CODING_NETEQ_COMFORT_NOISE_H_
