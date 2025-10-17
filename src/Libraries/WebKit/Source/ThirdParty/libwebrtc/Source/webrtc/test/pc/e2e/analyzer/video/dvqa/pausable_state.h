/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 13, 2025.
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
#ifndef TEST_PC_E2E_ANALYZER_VIDEO_DVQA_PAUSABLE_STATE_H_
#define TEST_PC_E2E_ANALYZER_VIDEO_DVQA_PAUSABLE_STATE_H_

#include <cstdint>
#include <vector>

#include "api/units/time_delta.h"
#include "api/units/timestamp.h"
#include "system_wrappers/include/clock.h"

namespace webrtc {

// Provides ability to pause and resume and tell at any point was state paused
// or active.
class PausableState {
 public:
  // Creates a state as active.
  explicit PausableState(Clock* clock) : clock_(clock) {}
  PausableState(const PausableState&) = delete;
  PausableState& operator=(const PausableState&) = delete;
  PausableState(PausableState&&) = default;
  PausableState& operator=(PausableState&&) = default;

  // Pauses current state. States MUST be active.
  //
  // Complexity: O(1)
  void Pause();

  // Activates current state. State MUST be paused.
  //
  // Complexity: O(1)
  void Resume();

  // Returns is state is paused right now.
  //
  // Complexity: O(1)
  bool IsPaused() const;

  // Returns if last event before `time` was "pause".
  //
  // Complexity: O(log(n))
  bool WasPausedAt(Timestamp time) const;

  // Returns if next event after `time` was "resume".
  //
  // Complexity: O(log(n))
  bool WasResumedAfter(Timestamp time) const;

  // Returns time of last event or plus infinity if no events happened.
  //
  // Complexity O(1)
  Timestamp GetLastEventTime() const;

  // Returns sum of durations during which state was active starting from
  // time `time`.
  //
  // Complexity O(n)
  TimeDelta GetActiveDurationFrom(Timestamp time) const;

 private:
  struct Event {
    Timestamp time;
    bool is_paused;
  };

  // Returns position in `events_` which has time:
  // 1. Most right of the equals
  // 2. The biggest which is smaller
  // 3. -1 otherwise (first time is bigger than `time`)
  int64_t GetPos(Timestamp time) const;

  Clock* clock_;

  std::vector<Event> events_;
};

}  // namespace webrtc

#endif  // TEST_PC_E2E_ANALYZER_VIDEO_DVQA_PAUSABLE_STATE_H_
