/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 10, 2022.
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
#ifndef MODULES_AUDIO_CODING_NETEQ_TOOLS_AUDIO_CHECKSUM_H_
#define MODULES_AUDIO_CODING_NETEQ_TOOLS_AUDIO_CHECKSUM_H_

#include <memory>
#include <string>

#include "modules/audio_coding/neteq/tools/audio_sink.h"
#include "rtc_base/buffer.h"
#include "rtc_base/message_digest.h"
#include "rtc_base/string_encode.h"
#include "rtc_base/system/arch.h"

namespace webrtc {
namespace test {

class AudioChecksum : public AudioSink {
 public:
  AudioChecksum()
      : checksum_(rtc::MessageDigestFactory::Create(rtc::DIGEST_MD5)),
        checksum_result_(checksum_->Size()),
        finished_(false) {}

  AudioChecksum(const AudioChecksum&) = delete;
  AudioChecksum& operator=(const AudioChecksum&) = delete;

  bool WriteArray(const int16_t* audio, size_t num_samples) override {
    if (finished_)
      return false;

#ifndef WEBRTC_ARCH_LITTLE_ENDIAN
#error "Big-endian gives a different checksum"
#endif
    checksum_->Update(audio, num_samples * sizeof(*audio));
    return true;
  }

  // Finalizes the computations, and returns the checksum.
  std::string Finish() {
    if (!finished_) {
      finished_ = true;
      checksum_->Finish(checksum_result_.data(), checksum_result_.size());
    }
    return rtc::hex_encode(checksum_result_);
  }

 private:
  std::unique_ptr<rtc::MessageDigest> checksum_;
  rtc::Buffer checksum_result_;
  bool finished_;
};

}  // namespace test
}  // namespace webrtc
#endif  // MODULES_AUDIO_CODING_NETEQ_TOOLS_AUDIO_CHECKSUM_H_
