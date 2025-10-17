/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 30, 2023.
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
#include "media/base/test_utils.h"

#include <cstdint>

#include "api/video/video_frame.h"
#include "api/video/video_source_interface.h"

namespace cricket {

cricket::StreamParams CreateSimStreamParams(
    const std::string& cname,
    const std::vector<uint32_t>& ssrcs) {
  cricket::StreamParams sp;
  cricket::SsrcGroup sg(cricket::kSimSsrcGroupSemantics, ssrcs);
  sp.ssrcs = ssrcs;
  sp.ssrc_groups.push_back(sg);
  sp.cname = cname;
  return sp;
}

// There should be an rtx_ssrc per ssrc.
cricket::StreamParams CreateSimWithRtxStreamParams(
    const std::string& cname,
    const std::vector<uint32_t>& ssrcs,
    const std::vector<uint32_t>& rtx_ssrcs) {
  cricket::StreamParams sp = CreateSimStreamParams(cname, ssrcs);
  for (size_t i = 0; i < ssrcs.size(); ++i) {
    sp.AddFidSsrc(ssrcs[i], rtx_ssrcs[i]);
  }
  return sp;
}

// There should be one fec ssrc per ssrc.
cricket::StreamParams CreatePrimaryWithFecFrStreamParams(
    const std::string& cname,
    uint32_t primary_ssrc,
    uint32_t flexfec_ssrc) {
  cricket::StreamParams sp;
  sp.ssrcs = {primary_ssrc};
  sp.cname = cname;
  sp.AddFecFrSsrc(primary_ssrc, flexfec_ssrc);
  return sp;
}

}  // namespace cricket
