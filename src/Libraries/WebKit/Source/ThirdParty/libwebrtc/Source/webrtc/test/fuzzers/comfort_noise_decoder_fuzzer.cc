/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 8, 2023.
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
#include <algorithm>

#include "api/array_view.h"
#include "modules/audio_coding/codecs/cng/webrtc_cng.h"
#include "rtc_base/buffer.h"
#include "test/fuzzers/fuzz_data_helper.h"

namespace webrtc {
namespace test {
namespace {

void FuzzOneInputTest(rtc::ArrayView<const uint8_t> data) {
  FuzzDataHelper fuzz_data(data);
  ComfortNoiseDecoder cng_decoder;

  while (1) {
    if (!fuzz_data.CanReadBytes(1))
      break;
    const uint8_t sid_frame_len = fuzz_data.Read<uint8_t>();
    auto sid_frame = fuzz_data.ReadByteArray(sid_frame_len);
    if (sid_frame.empty())
      break;
    cng_decoder.UpdateSid(sid_frame);
    if (!fuzz_data.CanReadBytes(3))
      break;
    constexpr bool kTrueOrFalse[] = {true, false};
    const bool new_period = fuzz_data.SelectOneOf(kTrueOrFalse);
    constexpr size_t kOutputSizes[] = {80, 160, 320, 480};
    const size_t output_size = fuzz_data.SelectOneOf(kOutputSizes);
    const size_t num_generate_calls =
        std::min(fuzz_data.Read<uint8_t>(), static_cast<uint8_t>(17));
    rtc::BufferT<int16_t> output(output_size);
    for (size_t i = 0; i < num_generate_calls; ++i) {
      cng_decoder.Generate(output, new_period);
    }
  }
}

}  // namespace
}  // namespace test

void FuzzOneInput(const uint8_t* data, size_t size) {
  if (size > 5000) {
    return;
  }
  test::FuzzOneInputTest(rtc::ArrayView<const uint8_t>(data, size));
}

}  // namespace webrtc
