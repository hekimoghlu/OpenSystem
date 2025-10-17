/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 11, 2025.
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
#include "rtc_base/buffer_queue.h"

#include <stdint.h>
#include <string.h>

#include <algorithm>

namespace rtc {

BufferQueue::BufferQueue(size_t capacity, size_t default_size)
    : capacity_(capacity), default_size_(default_size) {}

BufferQueue::~BufferQueue() {
  RTC_DCHECK_RUN_ON(&sequence_checker_);
  for (Buffer* buffer : queue_)
    delete buffer;
  for (Buffer* buffer : free_list_)
    delete buffer;
}

size_t BufferQueue::size() const {
  RTC_DCHECK_RUN_ON(&sequence_checker_);
  return queue_.size();
}

void BufferQueue::Clear() {
  RTC_DCHECK_RUN_ON(&sequence_checker_);
  while (!queue_.empty()) {
    free_list_.push_back(queue_.front());
    queue_.pop_front();
  }
}

bool BufferQueue::ReadFront(void* buffer, size_t bytes, size_t* bytes_read) {
  RTC_DCHECK_RUN_ON(&sequence_checker_);
  if (queue_.empty())
    return false;

  Buffer* packet = queue_.front();
  queue_.pop_front();

  bytes = std::min(bytes, packet->size());
  memcpy(buffer, packet->data(), bytes);

  if (bytes_read)
    *bytes_read = bytes;

  free_list_.push_back(packet);
  return true;
}

bool BufferQueue::WriteBack(const void* buffer,
                            size_t bytes,
                            size_t* bytes_written) {
  RTC_DCHECK_RUN_ON(&sequence_checker_);
  if (queue_.size() == capacity_)
    return false;

  Buffer* packet;
  if (!free_list_.empty()) {
    packet = free_list_.back();
    free_list_.pop_back();
  } else {
    packet = new Buffer(bytes, default_size_);
  }

  packet->SetData(static_cast<const uint8_t*>(buffer), bytes);
  if (bytes_written)
    *bytes_written = bytes;

  queue_.push_back(packet);
  return true;
}

}  // namespace rtc
