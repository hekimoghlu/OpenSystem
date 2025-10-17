/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 5, 2023.
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
#ifndef MODULES_VIDEO_CODING_H264_SPROP_PARAMETER_SETS_H_
#define MODULES_VIDEO_CODING_H264_SPROP_PARAMETER_SETS_H_

#include <cstdint>
#include <string>
#include <vector>

namespace webrtc {

class H264SpropParameterSets {
 public:
  H264SpropParameterSets() {}

  H264SpropParameterSets(const H264SpropParameterSets&) = delete;
  H264SpropParameterSets& operator=(const H264SpropParameterSets&) = delete;

  bool DecodeSprop(const std::string& sprop);
  const std::vector<uint8_t>& sps_nalu() { return sps_; }
  const std::vector<uint8_t>& pps_nalu() { return pps_; }

 private:
  std::vector<uint8_t> sps_;
  std::vector<uint8_t> pps_;
};

}  // namespace webrtc

#endif  // MODULES_VIDEO_CODING_H264_SPROP_PARAMETER_SETS_H_
