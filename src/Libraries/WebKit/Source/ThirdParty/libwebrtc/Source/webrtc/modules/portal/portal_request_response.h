/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 20, 2023.
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
#ifndef MODULES_PORTAL_PORTAL_REQUEST_RESPONSE_H_
#define MODULES_PORTAL_PORTAL_REQUEST_RESPONSE_H_

namespace webrtc {
namespace xdg_portal {

// Contains type of responses that can be observed when making a request to
// a desktop portal interface.
enum class RequestResponse {
  // Unknown, the initialized status.
  kUnknown,
  // Success, the request is carried out.
  kSuccess,
  // The user cancelled the interaction.
  kUserCancelled,
  // The user interaction was ended in some other way.
  kError,

  kMaxValue = kError,
};

}  // namespace xdg_portal
}  // namespace webrtc
#endif  // MODULES_PORTAL_PORTAL_REQUEST_RESPONSE_H_
