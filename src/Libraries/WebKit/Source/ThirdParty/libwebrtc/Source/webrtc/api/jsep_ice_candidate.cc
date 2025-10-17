/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 1, 2022.
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

#include <cstddef>
#include <memory>
#include <string>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/memory/memory.h"
#include "api/candidate.h"
#include "api/jsep.h"

namespace webrtc {

std::string JsepIceCandidate::sdp_mid() const {
  return sdp_mid_;
}

int JsepIceCandidate::sdp_mline_index() const {
  return sdp_mline_index_;
}

const cricket::Candidate& JsepIceCandidate::candidate() const {
  return candidate_;
}

std::string JsepIceCandidate::server_url() const {
  return candidate_.url();
}

JsepCandidateCollection::JsepCandidateCollection() = default;

JsepCandidateCollection::JsepCandidateCollection(JsepCandidateCollection&& o)
    : candidates_(std::move(o.candidates_)) {}

size_t JsepCandidateCollection::count() const {
  return candidates_.size();
}

void JsepCandidateCollection::add(JsepIceCandidate* candidate) {
  candidates_.push_back(absl::WrapUnique(candidate));
}

const IceCandidateInterface* JsepCandidateCollection::at(size_t index) const {
  return candidates_[index].get();
}

bool JsepCandidateCollection::HasCandidate(
    const IceCandidateInterface* candidate) const {
  return absl::c_any_of(
      candidates_, [&](const std::unique_ptr<JsepIceCandidate>& entry) {
        return entry->sdp_mid() == candidate->sdp_mid() &&
               entry->sdp_mline_index() == candidate->sdp_mline_index() &&
               entry->candidate().IsEquivalent(candidate->candidate());
      });
}

size_t JsepCandidateCollection::remove(const cricket::Candidate& candidate) {
  auto iter = absl::c_find_if(
      candidates_, [&](const std::unique_ptr<JsepIceCandidate>& c) {
        return candidate.MatchesForRemoval(c->candidate());
      });
  if (iter != candidates_.end()) {
    candidates_.erase(iter);
    return 1;
  }
  return 0;
}

}  // namespace webrtc
