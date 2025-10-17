/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 3, 2025.
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
#include "api/jsep.h"

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace webrtc {

std::string IceCandidateInterface::server_url() const {
  return "";
}

size_t SessionDescriptionInterface::RemoveCandidates(
    const std::vector<cricket::Candidate>& /* candidates */) {
  return 0;
}

const char SessionDescriptionInterface::kOffer[] = "offer";
const char SessionDescriptionInterface::kPrAnswer[] = "pranswer";
const char SessionDescriptionInterface::kAnswer[] = "answer";
const char SessionDescriptionInterface::kRollback[] = "rollback";

const char* SdpTypeToString(SdpType type) {
  switch (type) {
    case SdpType::kOffer:
      return SessionDescriptionInterface::kOffer;
    case SdpType::kPrAnswer:
      return SessionDescriptionInterface::kPrAnswer;
    case SdpType::kAnswer:
      return SessionDescriptionInterface::kAnswer;
    case SdpType::kRollback:
      return SessionDescriptionInterface::kRollback;
  }
  return "";
}

std::optional<SdpType> SdpTypeFromString(const std::string& type_str) {
  if (type_str == SessionDescriptionInterface::kOffer) {
    return SdpType::kOffer;
  } else if (type_str == SessionDescriptionInterface::kPrAnswer) {
    return SdpType::kPrAnswer;
  } else if (type_str == SessionDescriptionInterface::kAnswer) {
    return SdpType::kAnswer;
  } else if (type_str == SessionDescriptionInterface::kRollback) {
    return SdpType::kRollback;
  } else {
    return std::nullopt;
  }
}

}  // namespace webrtc
