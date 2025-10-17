/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 17, 2025.
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
#ifndef TEST_PC_E2E_PEER_PARAMS_PREPROCESSOR_H_
#define TEST_PC_E2E_PEER_PARAMS_PREPROCESSOR_H_

#include <memory>
#include <set>
#include <string>

#include "api/test/pclf/peer_configurer.h"

namespace webrtc {
namespace webrtc_pc_e2e {

class PeerParamsPreprocessor {
 public:
  PeerParamsPreprocessor();
  ~PeerParamsPreprocessor();

  // Set missing params to default values if it is required:
  //  * Generate video stream labels if some of them are missing
  //  * Generate audio stream labels if some of them are missing
  //  * Set video source generation mode if it is not specified
  //  * Video codecs under test
  void SetDefaultValuesForMissingParams(PeerConfigurer& peer);

  // Validate peer's parameters, also ensure uniqueness of all video stream
  // labels.
  void ValidateParams(const PeerConfigurer& peer);

 private:
  class DefaultNamesProvider;
  std::unique_ptr<DefaultNamesProvider> peer_names_provider_;

  std::set<std::string> peer_names_;
  std::set<std::string> video_labels_;
  std::set<std::string> audio_labels_;
  std::set<std::string> video_sync_groups_;
  std::set<std::string> audio_sync_groups_;
};

}  // namespace webrtc_pc_e2e
}  // namespace webrtc

#endif  // TEST_PC_E2E_PEER_PARAMS_PREPROCESSOR_H_
