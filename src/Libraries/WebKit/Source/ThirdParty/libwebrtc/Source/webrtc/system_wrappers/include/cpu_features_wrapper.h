/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 22, 2025.
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
#ifndef SYSTEM_WRAPPERS_INCLUDE_CPU_FEATURES_WRAPPER_H_
#define SYSTEM_WRAPPERS_INCLUDE_CPU_FEATURES_WRAPPER_H_

#include <stdint.h>

namespace webrtc {

// List of features in x86.
typedef enum { kSSE2, kSSE3, kAVX2, kFMA3 } CPUFeature;

// List of features in ARM.
enum {
  kCPUFeatureARMv7 = (1 << 0),
  kCPUFeatureVFPv3 = (1 << 1),
  kCPUFeatureNEON = (1 << 2),
  kCPUFeatureLDREXSTREX = (1 << 3)
};

// Returns true if the CPU supports the feature.
int GetCPUInfo(CPUFeature feature);

// No CPU feature is available => straight C path.
int GetCPUInfoNoASM(CPUFeature feature);

// Return the features in an ARM device.
// It detects the features in the hardware platform, and returns supported
// values in the above enum definition as a bitmask.
uint64_t GetCPUFeaturesARM(void);

}  // namespace webrtc

#endif  // SYSTEM_WRAPPERS_INCLUDE_CPU_FEATURES_WRAPPER_H_
