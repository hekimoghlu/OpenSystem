/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 7, 2025.
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
#ifndef TEST_FAKE_DECODER_H_
#define TEST_FAKE_DECODER_H_

#include <stdint.h>

#include <memory>

#include "api/task_queue/task_queue_base.h"
#include "api/task_queue/task_queue_factory.h"
#include "api/video/encoded_image.h"
#include "api/video_codecs/video_decoder.h"
#include "modules/video_coding/include/video_codec_interface.h"

namespace webrtc {
namespace test {

class FakeDecoder : public VideoDecoder {
 public:
  enum { kDefaultWidth = 320, kDefaultHeight = 180 };

  FakeDecoder();
  explicit FakeDecoder(TaskQueueFactory* task_queue_factory);
  virtual ~FakeDecoder() {}

  bool Configure(const Settings& settings) override;

  int32_t Decode(const EncodedImage& input,
                 int64_t render_time_ms) override;

  int32_t RegisterDecodeCompleteCallback(
      DecodedImageCallback* callback) override;

  int32_t Release() override;

  DecoderInfo GetDecoderInfo() const override;
  const char* ImplementationName() const override;

  static const char* kImplementationName;

  void SetDelayedDecoding(int decode_delay_ms);

 private:
  DecodedImageCallback* callback_;
  int width_;
  int height_;
  std::unique_ptr<TaskQueueBase, TaskQueueDeleter> task_queue_;
  TaskQueueFactory* task_queue_factory_;
  int decode_delay_ms_;
};

class FakeH264Decoder : public FakeDecoder {
 public:
  virtual ~FakeH264Decoder() {}

  int32_t Decode(const EncodedImage& input,
                 int64_t render_time_ms) override;
};

}  // namespace test
}  // namespace webrtc

#endif  // TEST_FAKE_DECODER_H_
