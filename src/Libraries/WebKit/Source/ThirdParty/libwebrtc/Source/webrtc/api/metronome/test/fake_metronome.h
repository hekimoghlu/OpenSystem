/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 16, 2024.
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
#ifndef API_METRONOME_TEST_FAKE_METRONOME_H_
#define API_METRONOME_TEST_FAKE_METRONOME_H_

#include <cstddef>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "api/metronome/metronome.h"
#include "api/units/time_delta.h"

namespace webrtc::test {

// ForcedTickMetronome is a Metronome that ticks when `Tick()` is invoked.
// The constructor argument `tick_period` returned in `TickPeriod()`.
class ForcedTickMetronome : public Metronome {
 public:
  explicit ForcedTickMetronome(TimeDelta tick_period);

  // Forces all TickListeners to run `OnTick`.
  void Tick();
  size_t NumListeners();

  // Metronome implementation.
  void RequestCallOnNextTick(absl::AnyInvocable<void() &&> callback) override;
  TimeDelta TickPeriod() const override;

 private:
  const TimeDelta tick_period_;
  std::vector<absl::AnyInvocable<void() &&>> callbacks_;
};

// FakeMetronome is a metronome that ticks based on a repeating task at the
// `tick_period` provided in the constructor. It is designed for use with
// simulated task queues for unit tests.
class FakeMetronome : public Metronome {
 public:
  explicit FakeMetronome(TimeDelta tick_period);

  void SetTickPeriod(TimeDelta tick_period);

  // Metronome implementation.
  void RequestCallOnNextTick(absl::AnyInvocable<void() &&> callback) override;
  TimeDelta TickPeriod() const override;

 private:
  TimeDelta tick_period_;
  std::vector<absl::AnyInvocable<void() &&>> callbacks_;
};

}  // namespace webrtc::test

#endif  // API_METRONOME_TEST_FAKE_METRONOME_H_
