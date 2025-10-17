/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 30, 2024.
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
#ifndef MEDIA_BASE_TEST_UTILS_H_
#define MEDIA_BASE_TEST_UTILS_H_

#include <string>
#include <vector>

#include "media/base/media_channel.h"
#include "media/base/video_common.h"
#include "rtc_base/arraysize.h"

namespace webrtc {
class VideoFrame;
}

namespace cricket {

// Returns size of 420 image with rounding on chroma for odd sizes.
#define I420_SIZE(w, h) (w * h + (((w + 1) / 2) * ((h + 1) / 2)) * 2)
// Returns size of ARGB image.
#define ARGB_SIZE(w, h) (w * h * 4)

template <class T>
inline std::vector<T> MakeVector(const T a[], size_t s) {
  return std::vector<T>(a, a + s);
}
#define MAKE_VECTOR(a) cricket::MakeVector(a, arraysize(a))

// Create Simulcast StreamParams with given `ssrcs` and `cname`.
cricket::StreamParams CreateSimStreamParams(const std::string& cname,
                                            const std::vector<uint32_t>& ssrcs);
// Create Simulcast stream with given `ssrcs` and `rtx_ssrcs`.
// The number of `rtx_ssrcs` must match number of `ssrcs`.
cricket::StreamParams CreateSimWithRtxStreamParams(
    const std::string& cname,
    const std::vector<uint32_t>& ssrcs,
    const std::vector<uint32_t>& rtx_ssrcs);

// Create StreamParams with single primary SSRC and corresponding FlexFEC SSRC.
cricket::StreamParams CreatePrimaryWithFecFrStreamParams(
    const std::string& cname,
    uint32_t primary_ssrc,
    uint32_t flexfec_ssrc);

}  // namespace cricket

#endif  // MEDIA_BASE_TEST_UTILS_H_
