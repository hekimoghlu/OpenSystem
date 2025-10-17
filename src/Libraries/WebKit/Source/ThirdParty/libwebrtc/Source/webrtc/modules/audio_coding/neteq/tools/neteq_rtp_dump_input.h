/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 17, 2025.
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
#ifndef MODULES_AUDIO_CODING_NETEQ_TOOLS_NETEQ_RTP_DUMP_INPUT_H_
#define MODULES_AUDIO_CODING_NETEQ_TOOLS_NETEQ_RTP_DUMP_INPUT_H_

#include <map>
#include <memory>
#include <optional>

#include "absl/strings/string_view.h"
#include "modules/audio_coding/neteq/tools/neteq_input.h"
#include "modules/rtp_rtcp/include/rtp_rtcp_defines.h"

namespace webrtc {
namespace test {

std::unique_ptr<NetEqInput> CreateNetEqRtpDumpInput(
    absl::string_view file_name,
    const std::map<int, RTPExtensionType>& hdr_ext_map,
    std::optional<uint32_t> ssrc_filter);

}  // namespace test
}  // namespace webrtc
#endif  // MODULES_AUDIO_CODING_NETEQ_TOOLS_NETEQ_RTP_DUMP_INPUT_H_
