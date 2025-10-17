/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 3, 2025.
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
#ifndef API_TEST_FAKE_FRAME_ENCRYPTOR_H_
#define API_TEST_FAKE_FRAME_ENCRYPTOR_H_

#include <stddef.h>
#include <stdint.h>

#include "api/array_view.h"
#include "api/crypto/frame_encryptor_interface.h"
#include "api/media_types.h"
#include "rtc_base/ref_counted_object.h"

namespace webrtc {

// The FakeFrameEncryptor is a TEST ONLY fake implementation of the
// FrameEncryptorInterface. It is constructed with a simple single digit key and
// a fixed postfix byte. This is just to validate that the core code works
// as expected.
class FakeFrameEncryptor
    : public rtc::RefCountedObject<FrameEncryptorInterface> {
 public:
  // Provide a key (0,255) and some postfix byte (0,255).
  explicit FakeFrameEncryptor(uint8_t fake_key = 0xAA,
                              uint8_t postfix_byte = 255);
  // Simply xors each payload with the provided fake key and adds the postfix
  // bit to the end. This will always fail if fail_encryption_ is set to true.
  int Encrypt(cricket::MediaType media_type,
              uint32_t ssrc,
              rtc::ArrayView<const uint8_t> additional_data,
              rtc::ArrayView<const uint8_t> frame,
              rtc::ArrayView<uint8_t> encrypted_frame,
              size_t* bytes_written) override;
  // Always returns 1 more than the size of the frame.
  size_t GetMaxCiphertextByteSize(cricket::MediaType media_type,
                                  size_t frame_size) override;
  // Sets the fake key to use during encryption.
  void SetFakeKey(uint8_t fake_key);
  // Returns the fake key used during encryption.
  uint8_t GetFakeKey() const;
  // Set the postfix byte to use.
  void SetPostfixByte(uint8_t expected_postfix_byte);
  // Return a postfix byte added to each outgoing payload.
  uint8_t GetPostfixByte() const;
  // Force all encryptions to fail.
  void SetFailEncryption(bool fail_encryption);

  enum class FakeEncryptionStatus : int {
    OK = 0,
    FORCED_FAILURE = 1,
  };

 private:
  uint8_t fake_key_ = 0;
  uint8_t postfix_byte_ = 0;
  bool fail_encryption_ = false;
};

}  // namespace webrtc

#endif  // API_TEST_FAKE_FRAME_ENCRYPTOR_H_
