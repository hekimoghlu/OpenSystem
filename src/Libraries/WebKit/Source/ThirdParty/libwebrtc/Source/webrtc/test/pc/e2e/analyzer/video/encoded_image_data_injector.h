/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 15, 2022.
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
#ifndef TEST_PC_E2E_ANALYZER_VIDEO_ENCODED_IMAGE_DATA_INJECTOR_H_
#define TEST_PC_E2E_ANALYZER_VIDEO_ENCODED_IMAGE_DATA_INJECTOR_H_

#include <cstdint>
#include <optional>
#include <utility>

#include "api/video/encoded_image.h"

namespace webrtc {
namespace webrtc_pc_e2e {

// Injects frame id into EncodedImage on encoder side
class EncodedImageDataInjector {
 public:
  virtual ~EncodedImageDataInjector() = default;

  // Return encoded image with specified `id` and `discard` flag injected into
  // its payload. `discard` flag mean does analyzing decoder should discard this
  // encoded image because it belongs to unnecessary simulcast stream or spatial
  // layer.
  virtual EncodedImage InjectData(uint16_t id,
                                  bool discard,
                                  const EncodedImage& source) = 0;
};

struct EncodedImageExtractionResult {
  std::optional<uint16_t> id;
  EncodedImage image;
  // Is true if encoded image should be discarded. It is used to filter out
  // unnecessary spatial layers and simulcast streams.
  bool discard;
};

// Extracts frame id from EncodedImage on decoder side.
class EncodedImageDataExtractor {
 public:
  virtual ~EncodedImageDataExtractor() = default;

  // Invoked by framework before any image will come to the extractor.
  // `expected_receivers_count` is the expected amount of receivers for each
  // encoded image.
  virtual void Start(int expected_receivers_count) = 0;

  // Invoked by framework when it is required to add one more receiver for
  // frames. Will be invoked before that receiver will start receive data.
  virtual void AddParticipantInCall() = 0;

  // Invoked by framework when it is required to remove receiver for frames.
  // Will be invoked after that receiver will stop receiving data.
  virtual void RemoveParticipantInCall() = 0;

  // Returns encoded image id, extracted from payload and also encoded image
  // with its original payload. For concatenated spatial layers it should be the
  // same id.
  virtual EncodedImageExtractionResult ExtractData(
      const EncodedImage& source) = 0;
};

class EncodedImageDataPropagator : public EncodedImageDataInjector,
                                   public EncodedImageDataExtractor {
 public:
  ~EncodedImageDataPropagator() override = default;
};

}  // namespace webrtc_pc_e2e
}  // namespace webrtc

#endif  // TEST_PC_E2E_ANALYZER_VIDEO_ENCODED_IMAGE_DATA_INJECTOR_H_
