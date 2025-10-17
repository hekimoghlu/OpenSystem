/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 29, 2022.
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
#ifndef COMMON_VIDEO_GENERIC_FRAME_DESCRIPTOR_GENERIC_FRAME_INFO_H_
#define COMMON_VIDEO_GENERIC_FRAME_DESCRIPTOR_GENERIC_FRAME_INFO_H_

#include <bitset>
#include <initializer_list>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/strings/string_view.h"
#include "api/transport/rtp/dependency_descriptor.h"
#include "api/video/video_codec_constants.h"
#include "rtc_base/system/rtc_export.h"

namespace webrtc {

// Describes how a certain encoder buffer was used when encoding a frame.
struct CodecBufferUsage {
  constexpr CodecBufferUsage(int id, bool referenced, bool updated)
      : id(id), referenced(referenced), updated(updated) {}

  int id = 0;
  bool referenced = false;
  bool updated = false;
};

struct RTC_EXPORT GenericFrameInfo : public FrameDependencyTemplate {
  class Builder;

  GenericFrameInfo();
  GenericFrameInfo(const GenericFrameInfo&);
  ~GenericFrameInfo();

  absl::InlinedVector<CodecBufferUsage, kMaxEncoderBuffers> encoder_buffers;
  std::vector<bool> part_of_chain;
  std::bitset<32> active_decode_targets = ~uint32_t{0};
};

class GenericFrameInfo::Builder {
 public:
  Builder();
  ~Builder();

  GenericFrameInfo Build() const;
  Builder& T(int temporal_id);
  Builder& S(int spatial_id);
  Builder& Dtis(absl::string_view indication_symbols);

 private:
  GenericFrameInfo info_;
};

}  // namespace webrtc

#endif  // COMMON_VIDEO_GENERIC_FRAME_DESCRIPTOR_GENERIC_FRAME_INFO_H_
