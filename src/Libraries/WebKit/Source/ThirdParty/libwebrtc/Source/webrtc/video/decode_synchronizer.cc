/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 18, 2023.
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
#include "video/decode_synchronizer.h"

#include <iterator>
#include <memory>
#include <utility>
#include <vector>

#include "api/sequence_checker.h"
#include "api/units/time_delta.h"
#include "api/units/timestamp.h"
#include "rtc_base/checks.h"
#include "rtc_base/logging.h"
#include "rtc_base/trace_event.h"
#include "video/frame_decode_scheduler.h"
#include "video/frame_decode_timing.h"

namespace webrtc {

DecodeSynchronizer::ScheduledFrame::ScheduledFrame(
    uint32_t rtp_timestamp,
    FrameDecodeTiming::FrameSchedule schedule,
    FrameDecodeScheduler::FrameReleaseCallback callback)
    : rtp_timestamp_(rtp_timestamp),
      schedule_(std::move(schedule)),
      callback_(std::move(callback)) {}

void DecodeSynchronizer::ScheduledFrame::RunFrameReleaseCallback() && {
  // Inspiration from Chromium base::OnceCallback. Move `*this` to a local
  // before execution to ensure internal state is cleared after callback
  // execution.
  auto sf = std::move(*this);
  std::move(sf.callback_)(sf.rtp_timestamp_, sf.schedule_.render_time);
}

Timestamp DecodeSynchronizer::ScheduledFrame::LatestDecodeTime() const {
  return schedule_.latest_decode_time;
}

DecodeSynchronizer::SynchronizedFrameDecodeScheduler::
    SynchronizedFrameDecodeScheduler(DecodeSynchronizer* sync)
    : sync_(sync) {
  RTC_DCHECK(sync_);
}

DecodeSynchronizer::SynchronizedFrameDecodeScheduler::
    ~SynchronizedFrameDecodeScheduler() {
  RTC_DCHECK(!next_frame_);
  RTC_DCHECK(stopped_);
}

std::optional<uint32_t>
DecodeSynchronizer::SynchronizedFrameDecodeScheduler::ScheduledRtpTimestamp() {
  return next_frame_.has_value()
             ? std::make_optional(next_frame_->rtp_timestamp())
             : std::nullopt;
}

DecodeSynchronizer::ScheduledFrame
DecodeSynchronizer::SynchronizedFrameDecodeScheduler::ReleaseNextFrame() {
  RTC_DCHECK(!stopped_);
  RTC_DCHECK(next_frame_);
  auto res = std::move(*next_frame_);
  next_frame_.reset();
  return res;
}

Timestamp
DecodeSynchronizer::SynchronizedFrameDecodeScheduler::LatestDecodeTime() {
  RTC_DCHECK(next_frame_);
  return next_frame_->LatestDecodeTime();
}

void DecodeSynchronizer::SynchronizedFrameDecodeScheduler::ScheduleFrame(
    uint32_t rtp,
    FrameDecodeTiming::FrameSchedule schedule,
    FrameReleaseCallback cb) {
  RTC_DCHECK(!stopped_);
  RTC_DCHECK(!next_frame_) << "Can not schedule two frames at once.";
  next_frame_ = ScheduledFrame(rtp, std::move(schedule), std::move(cb));
  sync_->OnFrameScheduled(this);
}

void DecodeSynchronizer::SynchronizedFrameDecodeScheduler::CancelOutstanding() {
  next_frame_.reset();
}

void DecodeSynchronizer::SynchronizedFrameDecodeScheduler::Stop() {
  if (stopped_) {
    return;
  }
  CancelOutstanding();
  stopped_ = true;
  sync_->RemoveFrameScheduler(this);
}

DecodeSynchronizer::DecodeSynchronizer(Clock* clock,
                                       Metronome* metronome,
                                       TaskQueueBase* worker_queue)
    : clock_(clock), worker_queue_(worker_queue), metronome_(metronome) {
  RTC_DCHECK(metronome_);
  RTC_DCHECK(worker_queue_);
}

DecodeSynchronizer::~DecodeSynchronizer() {
  RTC_DCHECK_RUN_ON(worker_queue_);
  RTC_CHECK(schedulers_.empty());
}

std::unique_ptr<FrameDecodeScheduler>
DecodeSynchronizer::CreateSynchronizedFrameScheduler() {
  TRACE_EVENT0("webrtc", __func__);
  RTC_DCHECK_RUN_ON(worker_queue_);
  auto scheduler = std::make_unique<SynchronizedFrameDecodeScheduler>(this);
  auto [it, inserted] = schedulers_.emplace(scheduler.get());
  // If this is the first `scheduler` added, start listening to the metronome.
  if (inserted && schedulers_.size() == 1) {
    RTC_DLOG(LS_VERBOSE) << "Listening to metronome";
    ScheduleNextTick();
  }

  return std::move(scheduler);
}

void DecodeSynchronizer::OnFrameScheduled(
    SynchronizedFrameDecodeScheduler* scheduler) {
  RTC_DCHECK_RUN_ON(worker_queue_);
  RTC_DCHECK(scheduler->ScheduledRtpTimestamp());

  Timestamp now = clock_->CurrentTime();
  Timestamp next_tick = expected_next_tick_;
  // If no tick has registered yet assume it will occur in the tick period.
  if (next_tick.IsInfinite()) {
    next_tick = now + metronome_->TickPeriod();
  }

  // Release the frame right away if the decode time is too soon. Otherwise
  // the stream may fall behind too much.
  bool decode_before_next_tick =
      scheduler->LatestDecodeTime() <
      (next_tick - FrameDecodeTiming::kMaxAllowedFrameDelay);
  // Decode immediately if the decode time is in the past.
  bool decode_time_in_past = scheduler->LatestDecodeTime() < now;

  if (decode_before_next_tick || decode_time_in_past) {
    ScheduledFrame scheduled_frame = scheduler->ReleaseNextFrame();
    std::move(scheduled_frame).RunFrameReleaseCallback();
  }
}

void DecodeSynchronizer::RemoveFrameScheduler(
    SynchronizedFrameDecodeScheduler* scheduler) {
  TRACE_EVENT0("webrtc", __func__);
  RTC_DCHECK_RUN_ON(worker_queue_);
  RTC_DCHECK(scheduler);
  auto it = schedulers_.find(scheduler);
  if (it == schedulers_.end()) {
    return;
  }
  schedulers_.erase(it);
  // If there are no more schedulers active, stop listening for metronome ticks.
  if (schedulers_.empty()) {
    expected_next_tick_ = Timestamp::PlusInfinity();
  }
}

void DecodeSynchronizer::ScheduleNextTick() {
  RTC_DCHECK_RUN_ON(worker_queue_);
  if (tick_scheduled_) {
    return;
  }
  tick_scheduled_ = true;
  metronome_->RequestCallOnNextTick(
      SafeTask(safety_.flag(), [this] { OnTick(); }));
}

void DecodeSynchronizer::OnTick() {
  TRACE_EVENT0("webrtc", __func__);
  RTC_DCHECK_RUN_ON(worker_queue_);
  tick_scheduled_ = false;
  expected_next_tick_ = clock_->CurrentTime() + metronome_->TickPeriod();

  for (auto* scheduler : schedulers_) {
    if (scheduler->ScheduledRtpTimestamp() &&
        scheduler->LatestDecodeTime() < expected_next_tick_) {
      auto scheduled_frame = scheduler->ReleaseNextFrame();
      std::move(scheduled_frame).RunFrameReleaseCallback();
    }
  }

  if (!schedulers_.empty())
    ScheduleNextTick();
}

}  // namespace webrtc
