/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 21, 2022.
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
#include "test/scenario/performance_stats.h"

#include <algorithm>

namespace webrtc {
namespace test {
void VideoFramesStats::AddFrameInfo(const VideoFrameBuffer& frame,
                                    Timestamp at_time) {
  ++count;
  RTC_DCHECK(at_time.IsFinite());
  pixels.AddSample(frame.width() * frame.height());
  resolution.AddSample(std::max(frame.width(), frame.height()));
  frames.AddEvent(at_time);
}

void VideoFramesStats::AddStats(const VideoFramesStats& other) {
  count += other.count;
  pixels.AddSamples(other.pixels);
  resolution.AddSamples(other.resolution);
  frames.AddEvents(other.frames);
}

void VideoQualityStats::AddStats(const VideoQualityStats& other) {
  capture.AddStats(other.capture);
  render.AddStats(other.render);
  lost_count += other.lost_count;
  freeze_count += other.freeze_count;
  capture_to_decoded_delay.AddSamples(other.capture_to_decoded_delay);
  end_to_end_delay.AddSamples(other.end_to_end_delay);
  psnr.AddSamples(other.psnr);
  psnr_with_freeze.AddSamples(other.psnr_with_freeze);
  skipped_between_rendered.AddSamples(other.skipped_between_rendered);
  freeze_duration.AddSamples(other.freeze_duration);
  time_between_freezes.AddSamples(other.time_between_freezes);
}

}  // namespace test
}  // namespace webrtc
