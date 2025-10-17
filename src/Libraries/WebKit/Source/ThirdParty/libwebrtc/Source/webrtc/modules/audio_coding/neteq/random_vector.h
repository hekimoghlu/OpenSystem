/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 1, 2024.
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
#ifndef MODULES_AUDIO_CODING_NETEQ_RANDOM_VECTOR_H_
#define MODULES_AUDIO_CODING_NETEQ_RANDOM_VECTOR_H_

#include <stddef.h>
#include <stdint.h>

namespace webrtc {

// This class generates pseudo-random samples.
class RandomVector {
 public:
  static const size_t kRandomTableSize = 256;
  static const int16_t kRandomTable[kRandomTableSize];

  RandomVector() : seed_(777), seed_increment_(1) {}

  RandomVector(const RandomVector&) = delete;
  RandomVector& operator=(const RandomVector&) = delete;

  void Reset();

  void Generate(size_t length, int16_t* output);

  void IncreaseSeedIncrement(int16_t increase_by);

  // Accessors and mutators.
  int16_t seed_increment() { return seed_increment_; }
  void set_seed_increment(int16_t value) { seed_increment_ = value; }

 private:
  uint32_t seed_;
  int16_t seed_increment_;
};

}  // namespace webrtc
#endif  // MODULES_AUDIO_CODING_NETEQ_RANDOM_VECTOR_H_
