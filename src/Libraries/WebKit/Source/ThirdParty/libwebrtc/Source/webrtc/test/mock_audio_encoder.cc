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
#include "test/mock_audio_encoder.h"

namespace webrtc {

MockAudioEncoder::MockAudioEncoder() = default;
MockAudioEncoder::~MockAudioEncoder() = default;

MockAudioEncoder::FakeEncoding::FakeEncoding(
    const AudioEncoder::EncodedInfo& info)
    : info_(info) {}

MockAudioEncoder::FakeEncoding::FakeEncoding(size_t encoded_bytes) {
  info_.encoded_bytes = encoded_bytes;
}

AudioEncoder::EncodedInfo MockAudioEncoder::FakeEncoding::operator()(
    uint32_t timestamp,
    rtc::ArrayView<const int16_t> audio,
    rtc::Buffer* encoded) {
  encoded->SetSize(encoded->size() + info_.encoded_bytes);
  return info_;
}

MockAudioEncoder::CopyEncoding::~CopyEncoding() = default;

MockAudioEncoder::CopyEncoding::CopyEncoding(
    AudioEncoder::EncodedInfo info,
    rtc::ArrayView<const uint8_t> payload)
    : info_(info), payload_(payload) {}

MockAudioEncoder::CopyEncoding::CopyEncoding(
    rtc::ArrayView<const uint8_t> payload)
    : payload_(payload) {
  info_.encoded_bytes = payload_.size();
}

AudioEncoder::EncodedInfo MockAudioEncoder::CopyEncoding::operator()(
    uint32_t timestamp,
    rtc::ArrayView<const int16_t> audio,
    rtc::Buffer* encoded) {
  RTC_CHECK(encoded);
  RTC_CHECK_LE(info_.encoded_bytes, payload_.size());
  encoded->AppendData(payload_.data(), info_.encoded_bytes);
  return info_;
}

}  // namespace webrtc
