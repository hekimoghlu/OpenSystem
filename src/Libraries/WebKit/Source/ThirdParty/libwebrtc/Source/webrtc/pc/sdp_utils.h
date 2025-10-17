/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 22, 2024.
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
#ifndef PC_SDP_UTILS_H_
#define PC_SDP_UTILS_H_

#include <functional>
#include <memory>
#include <string>

#include "api/jsep.h"
#include "p2p/base/transport_info.h"
#include "pc/session_description.h"
#include "rtc_base/system/rtc_export.h"

namespace webrtc {

// Returns a copy of the given session description.
RTC_EXPORT std::unique_ptr<SessionDescriptionInterface> CloneSessionDescription(
    const SessionDescriptionInterface* sdesc);

// Returns a copy of the given session description with the type changed.
RTC_EXPORT std::unique_ptr<SessionDescriptionInterface>
CloneSessionDescriptionAsType(const SessionDescriptionInterface* sdesc,
                              SdpType type);

// Function that takes a single session description content with its
// corresponding transport and produces a boolean.
typedef std::function<bool(const cricket::ContentInfo*,
                           const cricket::TransportInfo*)>
    SdpContentPredicate;

// Returns true if the predicate returns true for all contents in the given
// session description.
bool SdpContentsAll(SdpContentPredicate pred,
                    const cricket::SessionDescription* desc);

// Returns true if the predicate returns true for none of the contents in the
// given session description.
bool SdpContentsNone(SdpContentPredicate pred,
                     const cricket::SessionDescription* desc);

// Function that takes a single session description content with its
// corresponding transport and can mutate the content and/or the transport.
typedef std::function<void(cricket::ContentInfo*, cricket::TransportInfo*)>
    SdpContentMutator;

// Applies the mutator function over all contents in the given session
// description.
void SdpContentsForEach(SdpContentMutator fn,
                        cricket::SessionDescription* desc);

}  // namespace webrtc

#endif  // PC_SDP_UTILS_H_
