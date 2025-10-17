/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 23, 2023.
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
#include <stddef.h>
#include <stdint.h>

#include <vector>

#include "api/array_view.h"
#include "modules/rtp_rtcp/source/video_rtp_depacketizer_av1.h"
#include "test/fuzzers/fuzz_data_helper.h"

namespace webrtc {
void FuzzOneInput(const uint8_t* data, size_t size) {
  std::vector<rtc::ArrayView<const uint8_t>> rtp_payloads;

  // Convert plain array of bytes into array of array bytes.
  test::FuzzDataHelper fuzz_input(rtc::MakeArrayView(data, size));
  while (fuzz_input.CanReadBytes(sizeof(uint16_t))) {
    // In practice one rtp payload can be up to ~1200 - 1500 bytes. Majority
    // of the payload is just copied. To make fuzzing more efficient limit the
    // size of rtp payload to realistic value.
    uint16_t next_size = fuzz_input.Read<uint16_t>() % 1200;
    if (next_size > fuzz_input.BytesLeft()) {
      next_size = fuzz_input.BytesLeft();
    }
    rtp_payloads.push_back(fuzz_input.ReadByteArray(next_size));
  }
  // Run code under test.
  VideoRtpDepacketizerAv1().AssembleFrame(rtp_payloads);
}
}  // namespace webrtc
