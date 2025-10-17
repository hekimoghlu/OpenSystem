/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 17, 2022.
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
#ifndef API_SET_REMOTE_DESCRIPTION_OBSERVER_INTERFACE_H_
#define API_SET_REMOTE_DESCRIPTION_OBSERVER_INTERFACE_H_

#include "api/ref_count.h"
#include "api/rtc_error.h"

namespace webrtc {

// An observer for PeerConnectionInterface::SetRemoteDescription(). The
// callback is invoked such that the state of the peer connection can be
// examined to accurately reflect the effects of the SetRemoteDescription
// operation.
class SetRemoteDescriptionObserverInterface : public webrtc::RefCountInterface {
 public:
  // On success, `error.ok()` is true.
  virtual void OnSetRemoteDescriptionComplete(RTCError error) = 0;
};

}  // namespace webrtc

#endif  // API_SET_REMOTE_DESCRIPTION_OBSERVER_INTERFACE_H_
