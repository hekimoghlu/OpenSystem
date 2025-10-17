/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 26, 2022.
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
#include "pc/data_channel_utils.h"

#include <utility>

#include "rtc_base/checks.h"

namespace webrtc {

bool PacketQueue::Empty() const {
  return packets_.empty();
}

std::unique_ptr<DataBuffer> PacketQueue::PopFront() {
  RTC_DCHECK(!packets_.empty());
  byte_count_ -= packets_.front()->size();
  std::unique_ptr<DataBuffer> packet = std::move(packets_.front());
  packets_.pop_front();
  return packet;
}

void PacketQueue::PushFront(std::unique_ptr<DataBuffer> packet) {
  byte_count_ += packet->size();
  packets_.push_front(std::move(packet));
}

void PacketQueue::PushBack(std::unique_ptr<DataBuffer> packet) {
  byte_count_ += packet->size();
  packets_.push_back(std::move(packet));
}

void PacketQueue::Clear() {
  packets_.clear();
  byte_count_ = 0;
}

void PacketQueue::Swap(PacketQueue* other) {
  size_t other_byte_count = other->byte_count_;
  other->byte_count_ = byte_count_;
  byte_count_ = other_byte_count;

  other->packets_.swap(packets_);
}

}  // namespace webrtc
