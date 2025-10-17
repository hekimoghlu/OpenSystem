/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 6, 2023.
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
#ifndef MODULES_VIDEO_CODING_DEPRECATED_EVENT_WRAPPER_H_
#define MODULES_VIDEO_CODING_DEPRECATED_EVENT_WRAPPER_H_

namespace webrtc {
enum EventTypeWrapper { kEventSignaled = 1, kEventTimeout = 2 };

class EventWrapper {
 public:
  // Factory method. Constructor disabled.
  static EventWrapper* Create();

  virtual ~EventWrapper() {}

  // Releases threads who are calling Wait() and has started waiting. Please
  // note that a thread calling Wait() will not start waiting immediately.
  // assumptions to the contrary is a very common source of issues in
  // multithreaded programming.
  // Set is sticky in the sense that it will release at least one thread
  // either immediately or some time in the future.
  virtual bool Set() = 0;

  // Puts the calling thread into a wait state. The thread may be released
  // by a Set() call depending on if other threads are waiting and if so on
  // timing. The thread that was released will reset the event before leaving
  // preventing more threads from being released. If multiple threads
  // are waiting for the same Set(), only one (random) thread is guaranteed to
  // be released. It is possible that multiple (random) threads are released
  // Depending on timing.
  //
  // `max_time_ms` is the maximum time to wait in milliseconds.
  // TODO(bugs.webrtc.org/14366): Migrate to TimeDelta.
  virtual EventTypeWrapper Wait(int max_time_ms) = 0;
};

}  // namespace webrtc

#endif  // MODULES_VIDEO_CODING_DEPRECATED_EVENT_WRAPPER_H_
