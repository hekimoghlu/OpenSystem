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
#include "common_video/generic_frame_descriptor/generic_frame_info.h"

#include <utility>

#include "rtc_base/checks.h"

namespace webrtc {

GenericFrameInfo::GenericFrameInfo() = default;
GenericFrameInfo::GenericFrameInfo(const GenericFrameInfo&) = default;
GenericFrameInfo::~GenericFrameInfo() = default;

GenericFrameInfo::Builder::Builder() = default;
GenericFrameInfo::Builder::~Builder() = default;

GenericFrameInfo GenericFrameInfo::Builder::Build() const {
  return info_;
}

GenericFrameInfo::Builder& GenericFrameInfo::Builder::T(int temporal_id) {
  info_.temporal_id = temporal_id;
  return *this;
}

GenericFrameInfo::Builder& GenericFrameInfo::Builder::S(int spatial_id) {
  info_.spatial_id = spatial_id;
  return *this;
}

GenericFrameInfo::Builder& GenericFrameInfo::Builder::Dtis(
    absl::string_view indication_symbols) {
  info_.decode_target_indications =
      webrtc_impl::StringToDecodeTargetIndications(indication_symbols);
  return *this;
}

}  // namespace webrtc
