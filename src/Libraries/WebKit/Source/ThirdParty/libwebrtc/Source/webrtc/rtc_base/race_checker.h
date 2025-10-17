/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 26, 2022.
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
#ifndef RTC_BASE_RACE_CHECKER_H_
#define RTC_BASE_RACE_CHECKER_H_

#include "rtc_base/checks.h"
#include "rtc_base/platform_thread_types.h"
#include "rtc_base/thread_annotations.h"

namespace rtc {

namespace internal {
class RaceCheckerScope;
}  // namespace internal

// Best-effort race-checking implementation. This primitive uses no
// synchronization at all to be as-fast-as-possible in the non-racy case.
class RTC_LOCKABLE RaceChecker {
 public:
  friend class internal::RaceCheckerScope;
  RaceChecker();

 private:
  bool Acquire() const RTC_EXCLUSIVE_LOCK_FUNCTION();
  void Release() const RTC_UNLOCK_FUNCTION();

  // Volatile to prevent code being optimized away in Acquire()/Release().
  mutable volatile int access_count_ = 0;
  mutable volatile PlatformThreadRef accessing_thread_;
};

namespace internal {
class RTC_SCOPED_LOCKABLE RaceCheckerScope {
 public:
  explicit RaceCheckerScope(const RaceChecker* race_checker)
      RTC_EXCLUSIVE_LOCK_FUNCTION(race_checker);

  bool RaceDetected() const;
  ~RaceCheckerScope() RTC_UNLOCK_FUNCTION();

 private:
  const RaceChecker* const race_checker_;
  const bool race_check_ok_;
};

class RTC_SCOPED_LOCKABLE RaceCheckerScopeDoNothing {
 public:
  explicit RaceCheckerScopeDoNothing(const RaceChecker* race_checker)
      RTC_EXCLUSIVE_LOCK_FUNCTION(race_checker) {}

  ~RaceCheckerScopeDoNothing() RTC_UNLOCK_FUNCTION() {}
};

}  // namespace internal
}  // namespace rtc

#define RTC_CHECK_RUNS_SERIALIZED(x) RTC_CHECK_RUNS_SERIALIZED_NEXT(x, __LINE__)

#define RTC_CHECK_RUNS_SERIALIZED_NEXT(x, suffix) \
  RTC_CHECK_RUNS_SERIALIZED_IMPL(x, suffix)

#define RTC_CHECK_RUNS_SERIALIZED_IMPL(x, suffix)          \
  rtc::internal::RaceCheckerScope race_checker##suffix(x); \
  RTC_CHECK(!race_checker##suffix.RaceDetected())

#if RTC_DCHECK_IS_ON
#define RTC_DCHECK_RUNS_SERIALIZED(x)              \
  rtc::internal::RaceCheckerScope race_checker(x); \
  RTC_DCHECK(!race_checker.RaceDetected())
#else
#define RTC_DCHECK_RUNS_SERIALIZED(x) \
  rtc::internal::RaceCheckerScopeDoNothing race_checker(x)
#endif

#endif  // RTC_BASE_RACE_CHECKER_H_
