/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 28, 2024.
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
#ifndef API_CRYPTO_FRAME_DECRYPTOR_INTERFACE_H_
#define API_CRYPTO_FRAME_DECRYPTOR_INTERFACE_H_

#include <vector>

#include "api/array_view.h"
#include "api/media_types.h"
#include "rtc_base/ref_count.h"

namespace webrtc {

// FrameDecryptorInterface allows users to provide a custom decryption
// implementation for all incoming audio and video frames. The user must also
// provide a FrameEncryptorInterface to be able to encrypt the frames being
// sent out of the device. Note this is an additional layer of encyrption in
// addition to the standard SRTP mechanism and is not intended to be used
// without it. You may assume that this interface will have the same lifetime
// as the RTPReceiver it is attached to. It must only be attached to one
// RTPReceiver. Additional data may be null.
class FrameDecryptorInterface : public RefCountInterface {
 public:
  // The Status enum represents all possible states that can be
  // returned when attempting to decrypt a frame. kRecoverable indicates that
  // there was an error with the given frame and so it should not be passed to
  // the decoder, however it hints that the receive stream is still decryptable
  // which is important for determining when to send key frame requests
  // kUnknown should never be returned by the implementor.
  enum class Status { kOk, kRecoverable, kFailedToDecrypt, kUnknown };

  struct Result {
    Result(Status status, size_t bytes_written)
        : status(status), bytes_written(bytes_written) {}

    bool IsOk() const { return status == Status::kOk; }

    const Status status;
    const size_t bytes_written;
  };

  ~FrameDecryptorInterface() override {}

  // Attempts to decrypt the encrypted frame. You may assume the frame size will
  // be allocated to the size returned from GetMaxPlaintextSize. You may assume
  // that the frames are in order if SRTP is enabled. The stream is not provided
  // here and it is up to the implementor to transport this information to the
  // receiver if they care about it. You must set bytes_written to how many
  // bytes you wrote to in the frame buffer. kOk must be returned if successful,
  // kRecoverable should be returned if the failure was due to something other
  // than a decryption failure. kFailedToDecrypt should be returned in all other
  // cases.
  virtual Result Decrypt(cricket::MediaType media_type,
                         const std::vector<uint32_t>& csrcs,
                         rtc::ArrayView<const uint8_t> additional_data,
                         rtc::ArrayView<const uint8_t> encrypted_frame,
                         rtc::ArrayView<uint8_t> frame) = 0;

  // Returns the total required length in bytes for the output of the
  // decryption. This can be larger than the actual number of bytes you need but
  // must never be smaller as it informs the size of the frame buffer.
  virtual size_t GetMaxPlaintextByteSize(cricket::MediaType media_type,
                                         size_t encrypted_frame_size) = 0;
};

}  // namespace webrtc

#endif  // API_CRYPTO_FRAME_DECRYPTOR_INTERFACE_H_
