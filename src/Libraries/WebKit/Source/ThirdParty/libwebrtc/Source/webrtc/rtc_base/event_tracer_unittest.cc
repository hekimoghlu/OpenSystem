/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 12, 2024.
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
#include "rtc_base/event_tracer.h"

#include "rtc_base/synchronization/mutex.h"
#include "rtc_base/thread_annotations.h"
#include "rtc_base/trace_event.h"
#include "test/gtest.h"

namespace {

class TestStatistics {
 public:
  void Reset() {
    webrtc::MutexLock lock(&mutex_);
    events_logged_ = 0;
  }

  void Increment() {
    webrtc::MutexLock lock(&mutex_);
    ++events_logged_;
  }

  int Count() const {
    webrtc::MutexLock lock(&mutex_);
    return events_logged_;
  }

  static TestStatistics* Get() {
    // google.github.io/styleguide/cppguide.html#Static_and_Global_Variables
    static auto& test_stats = *new TestStatistics();
    return &test_stats;
  }

 private:
  mutable webrtc::Mutex mutex_;
  int events_logged_ RTC_GUARDED_BY(mutex_) = 0;
};

}  // namespace

namespace webrtc {

TEST(EventTracerTest, EventTracerDisabled) {
  { TRACE_EVENT0("webrtc-test", "EventTracerDisabled"); }
  EXPECT_FALSE(TestStatistics::Get()->Count());
  TestStatistics::Get()->Reset();
}

#if RTC_TRACE_EVENTS_ENABLED && !defined(RTC_USE_PERFETTO)
TEST(EventTracerTest, ScopedTraceEvent) {
  SetupEventTracer(
      [](const char* /*name*/) {
        return reinterpret_cast<const unsigned char*>("webrtc-test");
      },
      [](char /*phase*/, const unsigned char* /*category_enabled*/,
         const char* /*name*/, unsigned long long /*id*/, int /*num_args*/,
         const char** /*arg_names*/, const unsigned char* /*arg_types*/,
         const unsigned long long* /*arg_values*/,
         unsigned char /*flags*/) { TestStatistics::Get()->Increment(); });
  { TRACE_EVENT0("test", "ScopedTraceEvent"); }
  EXPECT_EQ(2, TestStatistics::Get()->Count());
  TestStatistics::Get()->Reset();
}
#endif

}  // namespace webrtc
