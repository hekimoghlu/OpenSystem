/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 10, 2023.
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
#ifndef PC_DATA_CHANNEL_UTILS_H_
#define PC_DATA_CHANNEL_UTILS_H_

#include <stddef.h>
#include <stdint.h>

#include <deque>
#include <memory>
#include <string>
#include <utility>

#include "api/data_channel_interface.h"
#include "media/base/media_engine.h"

namespace webrtc {

// A packet queue which tracks the total queued bytes. Queued packets are
// owned by this class.
class PacketQueue final {
 public:
  size_t byte_count() const { return byte_count_; }

  bool Empty() const;

  std::unique_ptr<DataBuffer> PopFront();

  void PushFront(std::unique_ptr<DataBuffer> packet);
  void PushBack(std::unique_ptr<DataBuffer> packet);

  void Clear();

  void Swap(PacketQueue* other);

 private:
  std::deque<std::unique_ptr<DataBuffer>> packets_;
  size_t byte_count_ = 0;
};

struct DataChannelStats {
  int internal_id;
  int id;
  std::string label;
  std::string protocol;
  DataChannelInterface::DataState state;
  uint32_t messages_sent;
  uint32_t messages_received;
  uint64_t bytes_sent;
  uint64_t bytes_received;
};

}  // namespace webrtc

#endif  // PC_DATA_CHANNEL_UTILS_H_
