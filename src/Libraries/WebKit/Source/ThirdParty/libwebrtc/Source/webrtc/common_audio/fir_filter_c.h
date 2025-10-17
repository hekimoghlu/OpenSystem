/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 24, 2024.
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
#ifndef COMMON_AUDIO_FIR_FILTER_C_H_
#define COMMON_AUDIO_FIR_FILTER_C_H_

#include <string.h>

#include <memory>

#include "common_audio/fir_filter.h"

namespace webrtc {

class FIRFilterC : public FIRFilter {
 public:
  FIRFilterC(const float* coefficients, size_t coefficients_length);
  ~FIRFilterC() override;

  void Filter(const float* in, size_t length, float* out) override;

 private:
  size_t coefficients_length_;
  size_t state_length_;
  std::unique_ptr<float[]> coefficients_;
  std::unique_ptr<float[]> state_;
};

}  // namespace webrtc

#endif  // COMMON_AUDIO_FIR_FILTER_C_H_
