/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 2, 2023.
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
#ifndef API_TEST_FAKE_FRAME_DECRYPTOR_H_
#define API_TEST_FAKE_FRAME_DECRYPTOR_H_

#include <stddef.h>
#include <stdint.h>

#include <vector>

#include "api/array_view.h"
#include "api/crypto/frame_decryptor_interface.h"
#include "api/media_types.h"

namespace webrtc {

// The FakeFrameDecryptor is a TEST ONLY fake implementation of the
// FrameDecryptorInterface. It is constructed with a simple single digit key and
// a fixed postfix byte. This is just to validate that the core code works
// as expected.
class FakeFrameDecryptor : public FrameDecryptorInterface {
 public:
  // Provide a key (0,255) and some postfix byte (0,255) this should match the
  // byte you expect from the FakeFrameEncryptor.
  explicit FakeFrameDecryptor(uint8_t fake_key = 0xAA,
                              uint8_t expected_postfix_byte = 255);
  // Fake decryption that just xors the payload with the 1 byte key and checks
  // the postfix byte. This will always fail if fail_decryption_ is set to true.
  Result Decrypt(cricket::MediaType media_type,
                 const std::vector<uint32_t>& csrcs,
                 rtc::ArrayView<const uint8_t> additional_data,
                 rtc::ArrayView<const uint8_t> encrypted_frame,
                 rtc::ArrayView<uint8_t> frame) override;
  // Always returns 1 less than the size of the encrypted frame.
  size_t GetMaxPlaintextByteSize(cricket::MediaType media_type,
                                 size_t encrypted_frame_size) override;
  // Sets the fake key to use for encryption.
  void SetFakeKey(uint8_t fake_key);
  // Returns the fake key used for encryption.
  uint8_t GetFakeKey() const;
  // Set the Postfix byte that is expected in the encrypted payload.
  void SetExpectedPostfixByte(uint8_t expected_postfix_byte);
  // Returns the postfix byte that will be checked for in the encrypted payload.
  uint8_t GetExpectedPostfixByte() const;
  // If set to true will force all encryption to fail.
  void SetFailDecryption(bool fail_decryption);
  // Simple error codes for tests to validate against.
  enum class FakeDecryptStatus : int {
    OK = 0,
    FORCED_FAILURE = 1,
    INVALID_POSTFIX = 2
  };

 private:
  uint8_t fake_key_ = 0;
  uint8_t expected_postfix_byte_ = 0;
  bool fail_decryption_ = false;
};

}  // namespace webrtc

#endif  // API_TEST_FAKE_FRAME_DECRYPTOR_H_
