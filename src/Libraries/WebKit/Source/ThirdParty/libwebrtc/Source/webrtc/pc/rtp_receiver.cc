/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 21, 2023.
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
#include "pc/rtp_receiver.h"

#include <stddef.h>

#include <atomic>
#include <string>
#include <utility>
#include <vector>

#include "api/media_stream_interface.h"
#include "api/scoped_refptr.h"
#include "pc/media_stream.h"
#include "pc/media_stream_proxy.h"
#include "rtc_base/thread.h"

namespace webrtc {

// This function is only expected to be called on the signalling thread.
// On the other hand, some test or even production setups may use
// several signaling threads.
int RtpReceiverInternal::GenerateUniqueId() {
  static std::atomic<int> g_unique_id{0};

  return ++g_unique_id;
}

std::vector<rtc::scoped_refptr<MediaStreamInterface>>
RtpReceiverInternal::CreateStreamsFromIds(std::vector<std::string> stream_ids) {
  std::vector<rtc::scoped_refptr<MediaStreamInterface>> streams(
      stream_ids.size());
  for (size_t i = 0; i < stream_ids.size(); ++i) {
    streams[i] = MediaStreamProxy::Create(
        rtc::Thread::Current(), MediaStream::Create(std::move(stream_ids[i])));
  }
  return streams;
}

}  // namespace webrtc
