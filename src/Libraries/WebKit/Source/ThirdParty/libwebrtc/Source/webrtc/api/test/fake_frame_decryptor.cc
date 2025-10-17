/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 27, 2023.
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
#include "api/test/fake_frame_decryptor.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "api/array_view.h"
#include "api/media_types.h"
#include "rtc_base/checks.h"

namespace webrtc {

FakeFrameDecryptor::FakeFrameDecryptor(uint8_t fake_key,
                                       uint8_t expected_postfix_byte)
    : fake_key_(fake_key), expected_postfix_byte_(expected_postfix_byte) {}

FakeFrameDecryptor::Result FakeFrameDecryptor::Decrypt(
    cricket::MediaType /* media_type */,
    const std::vector<uint32_t>& /* csrcs */,
    rtc::ArrayView<const uint8_t> /* additional_data */,
    rtc::ArrayView<const uint8_t> encrypted_frame,
    rtc::ArrayView<uint8_t> frame) {
  if (fail_decryption_) {
    return Result(Status::kFailedToDecrypt, 0);
  }

  RTC_CHECK_EQ(frame.size() + 1, encrypted_frame.size());
  for (size_t i = 0; i < frame.size(); i++) {
    frame[i] = encrypted_frame[i] ^ fake_key_;
  }

  if (encrypted_frame[frame.size()] != expected_postfix_byte_) {
    return Result(Status::kFailedToDecrypt, 0);
  }

  return Result(Status::kOk, frame.size());
}

size_t FakeFrameDecryptor::GetMaxPlaintextByteSize(
    cricket::MediaType /* media_type */,
    size_t encrypted_frame_size) {
  return encrypted_frame_size - 1;
}

void FakeFrameDecryptor::SetFakeKey(uint8_t fake_key) {
  fake_key_ = fake_key;
}

uint8_t FakeFrameDecryptor::GetFakeKey() const {
  return fake_key_;
}

void FakeFrameDecryptor::SetExpectedPostfixByte(uint8_t expected_postfix_byte) {
  expected_postfix_byte_ = expected_postfix_byte;
}

uint8_t FakeFrameDecryptor::GetExpectedPostfixByte() const {
  return expected_postfix_byte_;
}

void FakeFrameDecryptor::SetFailDecryption(bool fail_decryption) {
  fail_decryption_ = fail_decryption;
}

}  // namespace webrtc
