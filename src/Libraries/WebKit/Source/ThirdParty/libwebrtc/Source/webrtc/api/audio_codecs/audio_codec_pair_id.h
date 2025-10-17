/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 27, 2023.
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
#ifndef API_AUDIO_CODECS_AUDIO_CODEC_PAIR_ID_H_
#define API_AUDIO_CODECS_AUDIO_CODEC_PAIR_ID_H_

#include <stdint.h>

#include <utility>

namespace webrtc {

class AudioCodecPairId final {
 public:
  // Copyable, but not default constructible.
  AudioCodecPairId() = delete;
  AudioCodecPairId(const AudioCodecPairId&) = default;
  AudioCodecPairId(AudioCodecPairId&&) = default;
  AudioCodecPairId& operator=(const AudioCodecPairId&) = default;
  AudioCodecPairId& operator=(AudioCodecPairId&&) = default;

  friend void swap(AudioCodecPairId& a, AudioCodecPairId& b) {
    using std::swap;
    swap(a.id_, b.id_);
  }

  // Creates a new ID, unequal to any previously created ID.
  static AudioCodecPairId Create();

  // IDs can be tested for equality.
  friend bool operator==(AudioCodecPairId a, AudioCodecPairId b) {
    return a.id_ == b.id_;
  }
  friend bool operator!=(AudioCodecPairId a, AudioCodecPairId b) {
    return a.id_ != b.id_;
  }

  // Comparisons. The ordering of ID values is completely arbitrary, but
  // stable, so it's useful e.g. if you want to use IDs as keys in an ordered
  // map.
  friend bool operator<(AudioCodecPairId a, AudioCodecPairId b) {
    return a.id_ < b.id_;
  }
  friend bool operator<=(AudioCodecPairId a, AudioCodecPairId b) {
    return a.id_ <= b.id_;
  }
  friend bool operator>=(AudioCodecPairId a, AudioCodecPairId b) {
    return a.id_ >= b.id_;
  }
  friend bool operator>(AudioCodecPairId a, AudioCodecPairId b) {
    return a.id_ > b.id_;
  }

  // Returns a numeric representation of the ID. The numeric values are
  // completely arbitrary, but stable, collision-free, and reasonably evenly
  // distributed, so they are e.g. useful as hash values in unordered maps.
  uint64_t NumericRepresentation() const { return id_; }

 private:
  explicit AudioCodecPairId(uint64_t id) : id_(id) {}

  uint64_t id_;
};

}  // namespace webrtc

#endif  // API_AUDIO_CODECS_AUDIO_CODEC_PAIR_ID_H_
