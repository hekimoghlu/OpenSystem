/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 3, 2022.
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
#ifndef RTC_BASE_BUFFER_QUEUE_H_
#define RTC_BASE_BUFFER_QUEUE_H_

#include <stddef.h>

#include <deque>
#include <vector>

#include "api/sequence_checker.h"
#include "rtc_base/buffer.h"
#include "rtc_base/system/no_unique_address.h"
#include "rtc_base/thread_annotations.h"

namespace rtc {

class BufferQueue final {
 public:
  // Creates a buffer queue with a given capacity and default buffer size.
  BufferQueue(size_t capacity, size_t default_size);
  ~BufferQueue();

  BufferQueue(const BufferQueue&) = delete;
  BufferQueue& operator=(const BufferQueue&) = delete;

  // Return number of queued buffers.
  size_t size() const;

  // Clear the BufferQueue by moving all Buffers from `queue_` to `free_list_`.
  void Clear();

  // ReadFront will only read one buffer at a time and will truncate buffers
  // that don't fit in the passed memory.
  // Returns true unless no data could be returned.
  bool ReadFront(void* data, size_t bytes, size_t* bytes_read);

  // WriteBack always writes either the complete memory or nothing.
  // Returns true unless no data could be written.
  bool WriteBack(const void* data, size_t bytes, size_t* bytes_written);

  bool is_writable() const {
    RTC_DCHECK_RUN_ON(&sequence_checker_);
    return queue_.size() < capacity_;
  }

  bool is_readable() const {
    RTC_DCHECK_RUN_ON(&sequence_checker_);
    return !queue_.empty();
  }

 private:
  RTC_NO_UNIQUE_ADDRESS webrtc::SequenceChecker sequence_checker_;
  const size_t capacity_;
  const size_t default_size_;
  std::deque<Buffer*> queue_ RTC_GUARDED_BY(sequence_checker_);
  std::vector<Buffer*> free_list_ RTC_GUARDED_BY(sequence_checker_);
};

}  // namespace rtc

#endif  // RTC_BASE_BUFFER_QUEUE_H_
