/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 12, 2022.
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
#include "modules/rtp_rtcp/source/source_tracker.h"

#include <algorithm>
#include <utility>

#include "rtc_base/trace_event.h"

namespace webrtc {

SourceTracker::SourceTracker(Clock* clock) : clock_(clock) {
  RTC_DCHECK(clock_);
}

void SourceTracker::OnFrameDelivered(const RtpPacketInfos& packet_infos,
                                     Timestamp delivery_time) {
  TRACE_EVENT0("webrtc", "SourceTracker::OnFrameDelivered");
  if (packet_infos.empty()) {
    return;
  }
  if (delivery_time.IsInfinite()) {
    delivery_time = clock_->CurrentTime();
  }

  for (const RtpPacketInfo& packet_info : packet_infos) {
    for (uint32_t csrc : packet_info.csrcs()) {
      SourceKey key(RtpSourceType::CSRC, csrc);
      SourceEntry& entry = UpdateEntry(key);

      entry.timestamp = delivery_time;
      entry.audio_level = packet_info.audio_level();
      entry.absolute_capture_time = packet_info.absolute_capture_time();
      entry.local_capture_clock_offset =
          packet_info.local_capture_clock_offset();
      entry.rtp_timestamp = packet_info.rtp_timestamp();
    }

    SourceKey key(RtpSourceType::SSRC, packet_info.ssrc());
    SourceEntry& entry = UpdateEntry(key);

    entry.timestamp = delivery_time;
    entry.audio_level = packet_info.audio_level();
    entry.absolute_capture_time = packet_info.absolute_capture_time();
    entry.local_capture_clock_offset = packet_info.local_capture_clock_offset();
    entry.rtp_timestamp = packet_info.rtp_timestamp();
  }

  PruneEntries(delivery_time);
}

std::vector<RtpSource> SourceTracker::GetSources() const {
  PruneEntries(clock_->CurrentTime());

  std::vector<RtpSource> sources;
  for (const auto& pair : list_) {
    const SourceKey& key = pair.first;
    const SourceEntry& entry = pair.second;

    sources.emplace_back(
        entry.timestamp, key.source, key.source_type, entry.rtp_timestamp,
        RtpSource::Extensions{
            .audio_level = entry.audio_level,
            .absolute_capture_time = entry.absolute_capture_time,
            .local_capture_clock_offset = entry.local_capture_clock_offset});
  }

  return sources;
}

SourceTracker::SourceEntry& SourceTracker::UpdateEntry(const SourceKey& key) {
  // We intentionally do |find() + emplace()|, instead of checking the return
  // value of `emplace()`, for performance reasons. It's much more likely for
  // the key to already exist than for it not to.
  auto map_it = map_.find(key);
  if (map_it == map_.end()) {
    // Insert a new entry at the front of the list.
    list_.emplace_front(key, SourceEntry());
    map_.emplace(key, list_.begin());
  } else if (map_it->second != list_.begin()) {
    // Move the old entry to the front of the list.
    list_.splice(list_.begin(), list_, map_it->second);
  }

  return list_.front().second;
}

void SourceTracker::PruneEntries(Timestamp now) const {
  if (now < Timestamp::Zero() + kTimeout) {
    return;
  }
  Timestamp prune = now - kTimeout;
  while (!list_.empty() && list_.back().second.timestamp < prune) {
    map_.erase(list_.back().first);
    list_.pop_back();
  }
}

}  // namespace webrtc
