/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 30, 2024.
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
#ifndef COMMON_AUDIO_WINDOW_GENERATOR_H_
#define COMMON_AUDIO_WINDOW_GENERATOR_H_

#include <stddef.h>

namespace webrtc {

// Helper class with generators for various signal transform windows.
class WindowGenerator {
 public:
  WindowGenerator() = delete;
  WindowGenerator(const WindowGenerator&) = delete;
  WindowGenerator& operator=(const WindowGenerator&) = delete;

  static void Hanning(int length, float* window);
  static void KaiserBesselDerived(float alpha, size_t length, float* window);
};

}  // namespace webrtc

#endif  // COMMON_AUDIO_WINDOW_GENERATOR_H_
