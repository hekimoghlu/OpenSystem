/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 31, 2023.
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
#ifndef MODULES_AUDIO_CODING_NETEQ_TOOLS_NETEQ_PERFORMANCE_TEST_H_
#define MODULES_AUDIO_CODING_NETEQ_TOOLS_NETEQ_PERFORMANCE_TEST_H_

#include <stdint.h>

namespace webrtc {
namespace test {

class NetEqPerformanceTest {
 public:
  // Runs a performance test with parameters as follows:
  //   `runtime_ms`: the simulation time, i.e., the duration of the audio data.
  //   `lossrate`: drop one out of `lossrate` packets, e.g., one out of 10.
  //   `drift_factor`: clock drift in [0, 1].
  // Returns the runtime in ms.
  static int64_t Run(int runtime_ms, int lossrate, double drift_factor);
};

}  // namespace test
}  // namespace webrtc

#endif  // MODULES_AUDIO_CODING_NETEQ_TOOLS_NETEQ_PERFORMANCE_TEST_H_
