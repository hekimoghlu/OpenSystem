/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 28, 2025.
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
#ifndef API_CRYPTO_FRAME_ENCRYPTOR_INTERFACE_H_
#define API_CRYPTO_FRAME_ENCRYPTOR_INTERFACE_H_

#include "api/array_view.h"
#include "api/media_types.h"
#include "rtc_base/ref_count.h"

namespace webrtc {

// FrameEncryptorInterface allows users to provide a custom encryption
// implementation to encrypt all outgoing audio and video frames. The user must
// also provide a FrameDecryptorInterface to be able to decrypt the frames on
// the receiving device. Note this is an additional layer of encryption in
// addition to the standard SRTP mechanism and is not intended to be used
// without it. Implementations of this interface will have the same lifetime as
// the RTPSenders it is attached to. Additional data may be null.
class FrameEncryptorInterface : public RefCountInterface {
 public:
  ~FrameEncryptorInterface() override {}

  // Attempts to encrypt the provided frame. You may assume the encrypted_frame
  // will match the size returned by GetMaxCiphertextByteSize for a give frame.
  // You may assume that the frames will arrive in order if SRTP is enabled.
  // The ssrc will simply identify which stream the frame is travelling on. You
  // must set bytes_written to the number of bytes you wrote in the
  // encrypted_frame. 0 must be returned if successful all other numbers can be
  // selected by the implementer to represent error codes.
  virtual int Encrypt(cricket::MediaType media_type,
                      uint32_t ssrc,
                      rtc::ArrayView<const uint8_t> additional_data,
                      rtc::ArrayView<const uint8_t> frame,
                      rtc::ArrayView<uint8_t> encrypted_frame,
                      size_t* bytes_written) = 0;

  // Returns the total required length in bytes for the output of the
  // encryption. This can be larger than the actual number of bytes you need but
  // must never be smaller as it informs the size of the encrypted_frame buffer.
  virtual size_t GetMaxCiphertextByteSize(cricket::MediaType media_type,
                                          size_t frame_size) = 0;
};

}  // namespace webrtc

#endif  // API_CRYPTO_FRAME_ENCRYPTOR_INTERFACE_H_
