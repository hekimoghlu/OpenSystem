/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 5, 2022.
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
#ifndef CALL_ADAPTATION_DEGRADATION_PREFERENCE_PROVIDER_H_
#define CALL_ADAPTATION_DEGRADATION_PREFERENCE_PROVIDER_H_

#include "api/rtp_parameters.h"

namespace webrtc {

class DegradationPreferenceProvider {
 public:
  virtual ~DegradationPreferenceProvider();

  virtual DegradationPreference degradation_preference() const = 0;
};

}  // namespace webrtc

#endif  // CALL_ADAPTATION_DEGRADATION_PREFERENCE_PROVIDER_H_
