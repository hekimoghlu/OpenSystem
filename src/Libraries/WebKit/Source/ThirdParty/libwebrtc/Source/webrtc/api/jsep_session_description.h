/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 7, 2023.
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
// TODO(deadbeef): Move this out of api/; it's an implementation detail and
// shouldn't be used externally.

#ifndef API_JSEP_SESSION_DESCRIPTION_H_
#define API_JSEP_SESSION_DESCRIPTION_H_

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "api/candidate.h"
#include "api/jsep.h"
#include "api/jsep_ice_candidate.h"

namespace cricket {
class SessionDescription;
}

namespace webrtc {

// Implementation of SessionDescriptionInterface.
class JsepSessionDescription : public SessionDescriptionInterface {
 public:
  explicit JsepSessionDescription(SdpType type);
  // TODO(steveanton): Remove this once callers have switched to SdpType.
  explicit JsepSessionDescription(const std::string& type);
  JsepSessionDescription(
      SdpType type,
      std::unique_ptr<cricket::SessionDescription> description,
      absl::string_view session_id,
      absl::string_view session_version);
  virtual ~JsepSessionDescription();

  JsepSessionDescription(const JsepSessionDescription&) = delete;
  JsepSessionDescription& operator=(const JsepSessionDescription&) = delete;

  // Takes ownership of `description`.
  bool Initialize(std::unique_ptr<cricket::SessionDescription> description,
                  const std::string& session_id,
                  const std::string& session_version);

  virtual std::unique_ptr<SessionDescriptionInterface> Clone() const;

  virtual cricket::SessionDescription* description() {
    return description_.get();
  }
  virtual const cricket::SessionDescription* description() const {
    return description_.get();
  }
  virtual std::string session_id() const { return session_id_; }
  virtual std::string session_version() const { return session_version_; }
  virtual SdpType GetType() const { return type_; }
  virtual std::string type() const { return SdpTypeToString(type_); }
  // Allows changing the type. Used for testing.
  virtual bool AddCandidate(const IceCandidateInterface* candidate);
  virtual size_t RemoveCandidates(
      const std::vector<cricket::Candidate>& candidates);
  virtual size_t number_of_mediasections() const;
  virtual const IceCandidateCollection* candidates(
      size_t mediasection_index) const;
  virtual bool ToString(std::string* out) const;

 private:
  std::unique_ptr<cricket::SessionDescription> description_;
  std::string session_id_;
  std::string session_version_;
  SdpType type_;
  std::vector<JsepCandidateCollection> candidate_collection_;

  bool GetMediasectionIndex(const IceCandidateInterface* candidate,
                            size_t* index);
  int GetMediasectionIndex(const cricket::Candidate& candidate);
};

}  // namespace webrtc

#endif  // API_JSEP_SESSION_DESCRIPTION_H_
