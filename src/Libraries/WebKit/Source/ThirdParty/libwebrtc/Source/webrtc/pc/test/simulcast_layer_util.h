/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 15, 2023.
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
#ifndef PC_TEST_SIMULCAST_LAYER_UTIL_H_
#define PC_TEST_SIMULCAST_LAYER_UTIL_H_

#include <string>
#include <vector>

#include "api/jsep.h"
#include "api/rtp_transceiver_interface.h"
#include "pc/session_description.h"
#include "pc/simulcast_description.h"

namespace webrtc {

std::vector<cricket::SimulcastLayer> CreateLayers(
    const std::vector<std::string>& rids,
    const std::vector<bool>& active);

std::vector<cricket::SimulcastLayer> CreateLayers(
    const std::vector<std::string>& rids,
    bool active);

RtpTransceiverInit CreateTransceiverInit(
    const std::vector<cricket::SimulcastLayer>& layers);

cricket::SimulcastDescription RemoveSimulcast(SessionDescriptionInterface* sd);

}  // namespace webrtc

#endif  // PC_TEST_SIMULCAST_LAYER_UTIL_H_
