/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 1, 2022.
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
#include "api/test/fake_frame_encryptor.h"

#include <cstddef>
#include <cstdint>

#include "api/array_view.h"
#include "api/media_types.h"
#include "rtc_base/checks.h"

namespace webrtc {
FakeFrameEncryptor::FakeFrameEncryptor(uint8_t fake_key, uint8_t postfix_byte)
    : fake_key_(fake_key), postfix_byte_(postfix_byte) {}

// FrameEncryptorInterface implementation
int FakeFrameEncryptor::Encrypt(
    cricket::MediaType /* media_type */,
    uint32_t /* ssrc */,
    rtc::ArrayView<const uint8_t> /* additional_data */,
    rtc::ArrayView<const uint8_t> frame,
    rtc::ArrayView<uint8_t> encrypted_frame,
    size_t* bytes_written) {
  if (fail_encryption_) {
    return static_cast<int>(FakeEncryptionStatus::FORCED_FAILURE);
  }

  RTC_CHECK_EQ(frame.size() + 1, encrypted_frame.size());
  for (size_t i = 0; i < frame.size(); i++) {
    encrypted_frame[i] = frame[i] ^ fake_key_;
  }

  encrypted_frame[frame.size()] = postfix_byte_;
  *bytes_written = encrypted_frame.size();
  return static_cast<int>(FakeEncryptionStatus::OK);
}

size_t FakeFrameEncryptor::GetMaxCiphertextByteSize(
    cricket::MediaType /* media_type */,
    size_t frame_size) {
  return frame_size + 1;
}

void FakeFrameEncryptor::SetFakeKey(uint8_t fake_key) {
  fake_key_ = fake_key;
}

uint8_t FakeFrameEncryptor::GetFakeKey() const {
  return fake_key_;
}

void FakeFrameEncryptor::SetPostfixByte(uint8_t postfix_byte) {
  postfix_byte_ = postfix_byte;
}

uint8_t FakeFrameEncryptor::GetPostfixByte() const {
  return postfix_byte_;
}

void FakeFrameEncryptor::SetFailEncryption(bool fail_encryption) {
  fail_encryption_ = fail_encryption;
}

}  // namespace webrtc
