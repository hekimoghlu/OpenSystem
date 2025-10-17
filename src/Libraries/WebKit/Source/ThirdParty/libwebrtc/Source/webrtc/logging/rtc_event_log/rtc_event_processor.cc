/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 8, 2022.
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
#include "logging/rtc_event_log/rtc_event_processor.h"

#include <algorithm>
#include <cstdint>
#include <optional>

#include "rtc_base/numerics/sequence_number_util.h"

namespace webrtc {

RtcEventProcessor::RtcEventProcessor() = default;
RtcEventProcessor::~RtcEventProcessor() = default;

void RtcEventProcessor::ProcessEventsInOrder() {
  // `event_lists_` is a min-heap of lists ordered by the timestamp of the
  // first element in the list. We therefore process the first element of the
  // first list, then reinsert the remainder of that list into the heap
  // if the list still contains unprocessed elements.
  std::make_heap(event_lists_.begin(), event_lists_.end(), Cmp);

  while (!event_lists_.empty()) {
    event_lists_.front()->ProcessNext();
    std::pop_heap(event_lists_.begin(), event_lists_.end(), Cmp);
    if (event_lists_.back()->IsEmpty()) {
      event_lists_.pop_back();
    } else {
      std::push_heap(event_lists_.begin(), event_lists_.end(), Cmp);
    }
  }
}

bool RtcEventProcessor::Cmp(const RtcEventProcessor::ListPtrType& a,
                            const RtcEventProcessor::ListPtrType& b) {
  int64_t time_diff = a->GetNextTime() - b->GetNextTime();
  if (time_diff != 0)
    return time_diff > 0;

  if (a->GetTypeOrder() != b->GetTypeOrder())
    return a->GetTypeOrder() > b->GetTypeOrder();

  std::optional<uint16_t> wrapped_seq_num_a = a->GetTransportSeqNum();
  std::optional<uint16_t> wrapped_seq_num_b = b->GetTransportSeqNum();
  if (wrapped_seq_num_a && wrapped_seq_num_b) {
    return AheadOf<uint16_t>(*wrapped_seq_num_a, *wrapped_seq_num_b);
  } else if (wrapped_seq_num_a.has_value() != wrapped_seq_num_b.has_value()) {
    return wrapped_seq_num_a.has_value();
  }

  return a->GetInsertionOrder() > b->GetInsertionOrder();
}

}  // namespace webrtc
