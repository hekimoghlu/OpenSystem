/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 22, 2023.
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
#include "api/jsep_ice_candidate.h"

#include "pc/webrtc_sdp.h"

// This file contains JsepIceCandidate-related functions that are not
// included in api/jsep_ice_candidate.cc. Some of these link to SDP
// parsing/serializing functions, which some users may not want.
// TODO(bugs.webrtc.org/12330): Merge the two .cc files somehow.

namespace webrtc {

IceCandidateInterface* CreateIceCandidate(const std::string& sdp_mid,
                                          int sdp_mline_index,
                                          const std::string& sdp,
                                          SdpParseError* error) {
  JsepIceCandidate* jsep_ice = new JsepIceCandidate(sdp_mid, sdp_mline_index);
  if (!jsep_ice->Initialize(sdp, error)) {
    delete jsep_ice;
    return NULL;
  }
  return jsep_ice;
}

std::unique_ptr<IceCandidateInterface> CreateIceCandidate(
    const std::string& sdp_mid,
    int sdp_mline_index,
    const cricket::Candidate& candidate) {
  return std::make_unique<JsepIceCandidate>(sdp_mid, sdp_mline_index,
                                            candidate);
}

JsepIceCandidate::JsepIceCandidate(const std::string& sdp_mid,
                                   int sdp_mline_index)
    : sdp_mid_(sdp_mid), sdp_mline_index_(sdp_mline_index) {}

JsepIceCandidate::JsepIceCandidate(const std::string& sdp_mid,
                                   int sdp_mline_index,
                                   const cricket::Candidate& candidate)
    : sdp_mid_(sdp_mid),
      sdp_mline_index_(sdp_mline_index),
      candidate_(candidate) {}

JsepIceCandidate::~JsepIceCandidate() {}

JsepCandidateCollection JsepCandidateCollection::Clone() const {
  JsepCandidateCollection new_collection;
  for (const auto& candidate : candidates_) {
    new_collection.candidates_.push_back(std::make_unique<JsepIceCandidate>(
        candidate->sdp_mid(), candidate->sdp_mline_index(),
        candidate->candidate()));
  }
  return new_collection;
}

bool JsepIceCandidate::Initialize(const std::string& sdp, SdpParseError* err) {
  return SdpDeserializeCandidate(sdp, this, err);
}

bool JsepIceCandidate::ToString(std::string* out) const {
  if (!out)
    return false;
  *out = SdpSerializeCandidate(*this);
  return !out->empty();
}

}  // namespace webrtc
