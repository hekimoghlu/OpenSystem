/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 18, 2024.
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
#include "pc/sdp_utils.h"

#include <utility>
#include <vector>

#include "api/jsep_session_description.h"
#include "rtc_base/checks.h"

namespace webrtc {

std::unique_ptr<SessionDescriptionInterface> CloneSessionDescription(
    const SessionDescriptionInterface* sdesc) {
  RTC_DCHECK(sdesc);
  return CloneSessionDescriptionAsType(sdesc, sdesc->GetType());
}

std::unique_ptr<SessionDescriptionInterface> CloneSessionDescriptionAsType(
    const SessionDescriptionInterface* sdesc,
    SdpType type) {
  RTC_DCHECK(sdesc);
  auto clone = std::make_unique<JsepSessionDescription>(type);
  if (sdesc->description()) {
    clone->Initialize(sdesc->description()->Clone(), sdesc->session_id(),
                      sdesc->session_version());
  }
  // As of writing, our version of GCC does not allow returning a unique_ptr of
  // a subclass as a unique_ptr of a base class. To get around this, we need to
  // std::move the return value.
  return std::move(clone);
}

bool SdpContentsAll(SdpContentPredicate pred,
                    const cricket::SessionDescription* desc) {
  RTC_DCHECK(desc);
  for (const auto& content : desc->contents()) {
    const auto* transport_info = desc->GetTransportInfoByName(content.name);
    if (!pred(&content, transport_info)) {
      return false;
    }
  }
  return true;
}

bool SdpContentsNone(SdpContentPredicate pred,
                     const cricket::SessionDescription* desc) {
  return SdpContentsAll(
      [pred](const cricket::ContentInfo* content_info,
             const cricket::TransportInfo* transport_info) {
        return !pred(content_info, transport_info);
      },
      desc);
}

void SdpContentsForEach(SdpContentMutator fn,
                        cricket::SessionDescription* desc) {
  RTC_DCHECK(desc);
  for (auto& content : desc->contents()) {
    auto* transport_info = desc->GetTransportInfoByName(content.name);
    fn(&content, transport_info);
  }
}

}  // namespace webrtc
