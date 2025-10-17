/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 31, 2024.
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
#include "test/pc/e2e/analyzer/video/default_video_quality_analyzer_internal_shared_objects.h"

#include "api/video/video_frame.h"
#include "rtc_base/strings/string_builder.h"

namespace webrtc {

std::string InternalStatsKey::ToString() const {
  rtc::StringBuilder out;
  out << "stream=" << stream << "_sender=" << sender
      << "_receiver=" << receiver;
  return out.str();
}

bool operator<(const InternalStatsKey& a, const InternalStatsKey& b) {
  if (a.stream != b.stream) {
    return a.stream < b.stream;
  }
  if (a.sender != b.sender) {
    return a.sender < b.sender;
  }
  return a.receiver < b.receiver;
}

bool operator==(const InternalStatsKey& a, const InternalStatsKey& b) {
  return a.stream == b.stream && a.sender == b.sender &&
         a.receiver == b.receiver;
}

FrameComparison::FrameComparison(InternalStatsKey stats_key,
                                 std::optional<VideoFrame> captured,
                                 std::optional<VideoFrame> rendered,
                                 FrameComparisonType type,
                                 FrameStats frame_stats,
                                 OverloadReason overload_reason)
    : stats_key(std::move(stats_key)),
      captured(std::move(captured)),
      rendered(std::move(rendered)),
      type(type),
      frame_stats(std::move(frame_stats)),
      overload_reason(overload_reason) {}

}  // namespace webrtc
