/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 18, 2021.
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
#include "media/base/video_common.h"

#include <cstdint>
#include <numeric>
#include <string>

#include "api/array_view.h"
#include "rtc_base/arraysize.h"
#include "rtc_base/checks.h"
#include "rtc_base/strings/string_builder.h"

namespace cricket {

struct FourCCAliasEntry {
  uint32_t alias;
  uint32_t canonical;
};

static const FourCCAliasEntry kFourCCAliases[] = {
    {FOURCC_IYUV, FOURCC_I420},
    {FOURCC_YU16, FOURCC_I422},
    {FOURCC_YU24, FOURCC_I444},
    {FOURCC_YUYV, FOURCC_YUY2},
    {FOURCC_YUVS, FOURCC_YUY2},
    {FOURCC_HDYC, FOURCC_UYVY},
    {FOURCC_2VUY, FOURCC_UYVY},
    {FOURCC_JPEG, FOURCC_MJPG},  // Note: JPEG has DHT while MJPG does not.
    {FOURCC_DMB1, FOURCC_MJPG},
    {FOURCC_BA81, FOURCC_BGGR},
    {FOURCC_RGB3, FOURCC_RAW},
    {FOURCC_BGR3, FOURCC_24BG},
    {FOURCC_CM32, FOURCC_BGRA},
    {FOURCC_CM24, FOURCC_RAW},
};

uint32_t CanonicalFourCC(uint32_t fourcc) {
  for (uint32_t i = 0; i < arraysize(kFourCCAliases); ++i) {
    if (kFourCCAliases[i].alias == fourcc) {
      return kFourCCAliases[i].canonical;
    }
  }
  // Not an alias, so return it as-is.
  return fourcc;
}

// The C++ standard requires a namespace-scope definition of static const
// integral types even when they are initialized in the declaration (see
// [class.static.data]/4), but MSVC with /Ze is non-conforming and treats that
// as a multiply defined symbol error. See Also:
// http://msdn.microsoft.com/en-us/library/34h23df8.aspx
#ifndef _MSC_EXTENSIONS
const int64_t VideoFormat::kMinimumInterval;  // Initialized in header.
#endif

std::string VideoFormat::ToString() const {
  std::string fourcc_name = GetFourccName(fourcc) + " ";
  for (std::string::const_iterator i = fourcc_name.begin();
       i < fourcc_name.end(); ++i) {
    // Test character is printable; Avoid isprint() which asserts on negatives.
    if (*i < 32 || *i >= 127) {
      fourcc_name = "";
      break;
    }
  }

  char buf[256];
  rtc::SimpleStringBuilder sb(buf);
  sb << fourcc_name << width << "x" << height << "x"
     << IntervalToFpsFloat(interval);
  return sb.str();
}

int GreatestCommonDivisor(int a, int b) {
  RTC_DCHECK_GE(a, 0);
  RTC_DCHECK_GT(b, 0);
  return std::gcd(a, b);
}

int LeastCommonMultiple(int a, int b) {
  RTC_DCHECK_GT(a, 0);
  RTC_DCHECK_GT(b, 0);
  return std::lcm(a, b);
}

}  // namespace cricket
