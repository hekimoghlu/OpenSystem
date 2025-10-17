/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 31, 2025.
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
#include "test/pc/e2e/analyzer/video/dvqa/pausable_state.h"

#include <cstdint>

#include "api/units/time_delta.h"
#include "api/units/timestamp.h"
#include "rtc_base/checks.h"

namespace webrtc {

void PausableState::Pause() {
  if (!IsPaused()) {
    events_.push_back(Event{.time = clock_->CurrentTime(), .is_paused = true});
  }
}

void PausableState::Resume() {
  if (IsPaused()) {
    events_.push_back(Event{.time = clock_->CurrentTime(), .is_paused = false});
  }
}

bool PausableState::IsPaused() const {
  return !events_.empty() && events_.back().is_paused;
}

bool PausableState::WasPausedAt(Timestamp time) const {
  if (events_.empty()) {
    return false;
  }

  int64_t pos = GetPos(time);
  return pos != -1 && events_[pos].is_paused;
}

bool PausableState::WasResumedAfter(Timestamp time) const {
  if (events_.empty()) {
    return false;
  }

  int64_t pos = GetPos(time);
  return (pos + 1 < static_cast<int64_t>(events_.size())) &&
         !events_[pos + 1].is_paused;
}

Timestamp PausableState::GetLastEventTime() const {
  if (events_.empty()) {
    return Timestamp::PlusInfinity();
  }

  return events_.back().time;
}

TimeDelta PausableState::GetActiveDurationFrom(Timestamp time) const {
  if (events_.empty()) {
    return clock_->CurrentTime() - time;
  }

  int64_t pos = GetPos(time);
  TimeDelta duration = TimeDelta::Zero();
  for (int64_t i = pos; i < static_cast<int64_t>(events_.size()); ++i) {
    if (i == -1 || !events_[i].is_paused) {
      Timestamp start_time = (i == pos) ? time : events_[i].time;
      Timestamp end_time = (i + 1 == static_cast<int64_t>(events_.size()))
                               ? clock_->CurrentTime()
                               : events_[i + 1].time;

      duration += end_time - start_time;
    }
  }
  return duration;
}

int64_t PausableState::GetPos(Timestamp time) const {
  int64_t l = 0, r = events_.size() - 1;
  while (l < r) {
    int64_t pos = (l + r) / 2;
    if (time < events_[pos].time) {
      r = pos;
    } else if (time >= events_[pos].time) {
      l = pos + 1;
    }
  }
  if (time < events_[l].time) {
    return l - 1;
  } else {
    return l;
  }
}

}  // namespace webrtc
