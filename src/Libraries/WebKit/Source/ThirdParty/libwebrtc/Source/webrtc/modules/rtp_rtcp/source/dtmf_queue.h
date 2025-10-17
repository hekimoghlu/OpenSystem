/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 25, 2022.
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
#ifndef MODULES_RTP_RTCP_SOURCE_DTMF_QUEUE_H_
#define MODULES_RTP_RTCP_SOURCE_DTMF_QUEUE_H_

#include <stdint.h>

#include <list>

#include "rtc_base/synchronization/mutex.h"

namespace webrtc {
class DtmfQueue {
 public:
  struct Event {
    uint16_t duration_ms = 0;
    uint8_t payload_type = 0;
    uint8_t key = 0;
    uint8_t level = 0;
  };

  DtmfQueue();
  ~DtmfQueue();

  bool AddDtmf(const Event& event);
  bool NextDtmf(Event* event);
  bool PendingDtmf() const;

 private:
  mutable Mutex dtmf_mutex_;
  std::list<Event> queue_;
};
}  // namespace webrtc

#endif  // MODULES_RTP_RTCP_SOURCE_DTMF_QUEUE_H_
