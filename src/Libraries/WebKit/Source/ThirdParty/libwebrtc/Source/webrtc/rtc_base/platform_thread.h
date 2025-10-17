/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 12, 2025.
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
#ifndef RTC_BASE_PLATFORM_THREAD_H_
#define RTC_BASE_PLATFORM_THREAD_H_

#include <functional>
#include <string>
#if !defined(WEBRTC_WIN)
#include <pthread.h>
#endif

#include <optional>

#include "absl/strings/string_view.h"
#include "rtc_base/platform_thread_types.h"

namespace rtc {

enum class ThreadPriority {
  kLow = 1,
  kNormal,
  kHigh,
  kRealtime,
};

struct ThreadAttributes {
  ThreadPriority priority = ThreadPriority::kNormal;
  ThreadAttributes& SetPriority(ThreadPriority priority_param) {
    priority = priority_param;
    return *this;
  }
};

// Represents a simple worker thread.
class PlatformThread final {
 public:
  // Handle is the base platform thread handle.
#if defined(WEBRTC_WIN)
  using Handle = HANDLE;
#else
  using Handle = pthread_t;
#endif  // defined(WEBRTC_WIN)
  // This ctor creates the PlatformThread with an unset handle (returning true
  // in empty()) and is provided for convenience.
  // TODO(bugs.webrtc.org/12727) Look into if default and move support can be
  // removed.
  PlatformThread() = default;

  // Moves `rhs` into this, storing an empty state in `rhs`.
  // TODO(bugs.webrtc.org/12727) Look into if default and move support can be
  // removed.
  PlatformThread(PlatformThread&& rhs);

  // Copies won't work since we'd have problems with joinable threads.
  PlatformThread(const PlatformThread&) = delete;
  PlatformThread& operator=(const PlatformThread&) = delete;

  // Moves `rhs` into this, storing an empty state in `rhs`.
  // TODO(bugs.webrtc.org/12727) Look into if default and move support can be
  // removed.
  PlatformThread& operator=(PlatformThread&& rhs);

  // For a PlatformThread that's been spawned joinable, the destructor suspends
  // the calling thread until the created thread exits unless the thread has
  // already exited.
  virtual ~PlatformThread();

  // Finalizes any allocated resources.
  // For a PlatformThread that's been spawned joinable, Finalize() suspends
  // the calling thread until the created thread exits unless the thread has
  // already exited.
  // empty() returns true after completion.
  void Finalize();

  // Returns true if default constructed, moved from, or Finalize()ed.
  bool empty() const { return !handle_.has_value(); }

  // Creates a started joinable thread which will be joined when the returned
  // PlatformThread destructs or Finalize() is called.
  static PlatformThread SpawnJoinable(
      std::function<void()> thread_function,
      absl::string_view name,
      ThreadAttributes attributes = ThreadAttributes());

  // Creates a started detached thread. The caller has to use external
  // synchronization as nothing is provided by the PlatformThread construct.
  static PlatformThread SpawnDetached(
      std::function<void()> thread_function,
      absl::string_view name,
      ThreadAttributes attributes = ThreadAttributes());

  // Returns the base platform thread handle of this thread.
  std::optional<Handle> GetHandle() const;

#if defined(WEBRTC_WIN)
  // Queue a Windows APC function that runs when the thread is alertable.
  bool QueueAPC(PAPCFUNC apc_function, ULONG_PTR data);
#endif

 private:
  PlatformThread(Handle handle, bool joinable);
  static PlatformThread SpawnThread(std::function<void()> thread_function,
                                    absl::string_view name,
                                    ThreadAttributes attributes,
                                    bool joinable);

  std::optional<Handle> handle_;
  bool joinable_ = false;
};

}  // namespace rtc

#endif  // RTC_BASE_PLATFORM_THREAD_H_
