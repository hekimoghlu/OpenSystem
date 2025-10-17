/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 26, 2025.
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
#include "media/base/rid_description.h"

namespace cricket {

RidDescription::RidDescription() = default;
RidDescription::RidDescription(const std::string& rid, RidDirection direction)
    : rid{rid}, direction{direction} {}
RidDescription::RidDescription(const RidDescription& other) = default;
RidDescription::~RidDescription() = default;
RidDescription& RidDescription::operator=(const RidDescription& other) =
    default;
bool RidDescription::operator==(const RidDescription& other) const {
  return rid == other.rid && direction == other.direction &&
         payload_types == other.payload_types &&
         restrictions == other.restrictions;
}

}  // namespace cricket
