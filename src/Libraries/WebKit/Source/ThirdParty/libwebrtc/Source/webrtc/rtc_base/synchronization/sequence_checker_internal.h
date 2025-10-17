/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 27, 2023.
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
#ifndef RTC_BASE_SYNCHRONIZATION_SEQUENCE_CHECKER_INTERNAL_H_
#define RTC_BASE_SYNCHRONIZATION_SEQUENCE_CHECKER_INTERNAL_H_

#include <string>
#include <type_traits>

#include "api/task_queue/task_queue_base.h"
#include "rtc_base/platform_thread_types.h"
#include "rtc_base/synchronization/mutex.h"
#include "rtc_base/system/rtc_export.h"
#include "rtc_base/thread_annotations.h"

namespace webrtc {
namespace webrtc_sequence_checker_internal {

// Real implementation of SequenceChecker, for use in debug mode, or
// for temporary use in release mode (e.g. to RTC_CHECK on a threading issue
// seen only in the wild).
//
// Note: You should almost always use the SequenceChecker class to get the
// right version for your build configuration.
class RTC_EXPORT SequenceCheckerImpl {
 public:
  explicit SequenceCheckerImpl(bool attach_to_current_thread);
  explicit SequenceCheckerImpl(TaskQueueBase* attached_queue);
  ~SequenceCheckerImpl() = default;

  bool IsCurrent() const;
  // Changes the task queue or thread that is checked for in IsCurrent. This can
  // be useful when an object may be created on one task queue / thread and then
  // used exclusively on another thread.
  void Detach();

  // Returns a string that is formatted to match with the error string printed
  // by RTC_CHECK() when a condition is not met.
  // This is used in conjunction with the RTC_DCHECK_RUN_ON() macro.
  std::string ExpectationToString() const;

 private:
  mutable Mutex lock_;
  // These are mutable so that IsCurrent can set them.
  mutable bool attached_ RTC_GUARDED_BY(lock_);
  mutable rtc::PlatformThreadRef valid_thread_ RTC_GUARDED_BY(lock_);
  mutable const TaskQueueBase* valid_queue_ RTC_GUARDED_BY(lock_);
};

// Do nothing implementation, for use in release mode.
//
// Note: You should almost always use the SequenceChecker class to get the
// right version for your build configuration.
class SequenceCheckerDoNothing {
 public:
  explicit SequenceCheckerDoNothing(bool /* attach_to_current_thread */) {}
  explicit SequenceCheckerDoNothing(TaskQueueBase* /* attached_queue */) {}
  bool IsCurrent() const { return true; }
  void Detach() {}
};

template <typename ThreadLikeObject>
std::enable_if_t<std::is_base_of_v<SequenceCheckerImpl, ThreadLikeObject>,
                 std::string>
ExpectationToString([[maybe_unused]] const ThreadLikeObject* checker) {
#if RTC_DCHECK_IS_ON
  return checker->ExpectationToString();
#else
  return std::string();
#endif
}

// Catch-all implementation for types other than explicitly supported above.
template <typename ThreadLikeObject>
std::enable_if_t<!std::is_base_of_v<SequenceCheckerImpl, ThreadLikeObject>,
                 std::string>
ExpectationToString(const ThreadLikeObject*) {
  return std::string();
}

}  // namespace webrtc_sequence_checker_internal
}  // namespace webrtc

#endif  // RTC_BASE_SYNCHRONIZATION_SEQUENCE_CHECKER_INTERNAL_H_
