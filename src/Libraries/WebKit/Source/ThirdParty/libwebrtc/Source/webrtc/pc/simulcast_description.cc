/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 7, 2022.
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
#include "pc/simulcast_description.h"

#include "rtc_base/checks.h"

namespace cricket {

SimulcastLayer::SimulcastLayer(absl::string_view rid, bool is_paused)
    : rid{rid}, is_paused{is_paused} {
  RTC_DCHECK(!rid.empty());
}

bool SimulcastLayer::operator==(const SimulcastLayer& other) const {
  return rid == other.rid && is_paused == other.is_paused;
}

void SimulcastLayerList::AddLayer(const SimulcastLayer& layer) {
  list_.push_back({layer});
}

void SimulcastLayerList::AddLayerWithAlternatives(
    const std::vector<SimulcastLayer>& rids) {
  RTC_DCHECK(!rids.empty());
  list_.push_back(rids);
}

const std::vector<SimulcastLayer>& SimulcastLayerList::operator[](
    size_t index) const {
  RTC_DCHECK_LT(index, list_.size());
  return list_[index];
}

bool SimulcastDescription::empty() const {
  return send_layers_.empty() && receive_layers_.empty();
}

std::vector<SimulcastLayer> SimulcastLayerList::GetAllLayers() const {
  std::vector<SimulcastLayer> result;
  for (auto groupIt = begin(); groupIt != end(); groupIt++) {
    for (auto it = groupIt->begin(); it != groupIt->end(); it++) {
      result.push_back(*it);
    }
  }

  return result;
}

}  // namespace cricket
