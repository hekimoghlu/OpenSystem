/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 27, 2023.
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
#include "pc/test/simulcast_layer_util.h"

#include "absl/algorithm/container.h"
#include "rtc_base/checks.h"

namespace webrtc {

std::vector<cricket::SimulcastLayer> CreateLayers(
    const std::vector<std::string>& rids,
    const std::vector<bool>& active) {
  RTC_DCHECK_EQ(rids.size(), active.size());
  std::vector<cricket::SimulcastLayer> result;
  absl::c_transform(rids, active, std::back_inserter(result),
                    [](const std::string& rid, bool is_active) {
                      return cricket::SimulcastLayer(rid, !is_active);
                    });
  return result;
}

std::vector<cricket::SimulcastLayer> CreateLayers(
    const std::vector<std::string>& rids,
    bool active) {
  return CreateLayers(rids, std::vector<bool>(rids.size(), active));
}

RtpTransceiverInit CreateTransceiverInit(
    const std::vector<cricket::SimulcastLayer>& layers) {
  RtpTransceiverInit init;
  for (const cricket::SimulcastLayer& layer : layers) {
    RtpEncodingParameters encoding;
    encoding.rid = layer.rid;
    encoding.active = !layer.is_paused;
    init.send_encodings.push_back(encoding);
  }
  return init;
}

cricket::SimulcastDescription RemoveSimulcast(SessionDescriptionInterface* sd) {
  auto mcd = sd->description()->contents()[0].media_description();
  auto result = mcd->simulcast_description();
  mcd->set_simulcast_description(cricket::SimulcastDescription());
  return result;
}

}  // namespace webrtc
