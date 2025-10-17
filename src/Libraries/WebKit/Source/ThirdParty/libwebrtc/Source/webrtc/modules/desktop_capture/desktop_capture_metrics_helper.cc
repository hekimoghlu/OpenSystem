/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 18, 2022.
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
#include "modules/desktop_capture/desktop_capture_metrics_helper.h"

#include "modules/desktop_capture/desktop_capture_types.h"
#include "system_wrappers/include/metrics.h"

namespace webrtc {
namespace {
// This enum is logged via UMA so entries should not be reordered or have their
// values changed. This should also be kept in sync with the values in the
// DesktopCapturerId namespace.
enum class SequentialDesktopCapturerId {
  kUnknown = 0,
  kWgcCapturerWin = 1,
  // kScreenCapturerWinMagnifier = 2,
  kWindowCapturerWinGdi = 3,
  kScreenCapturerWinGdi = 4,
  kScreenCapturerWinDirectx = 5,
  kMaxValue = kScreenCapturerWinDirectx
};
}  // namespace

void RecordCapturerImpl(uint32_t capturer_id) {
  SequentialDesktopCapturerId sequential_id;
  switch (capturer_id) {
    case DesktopCapturerId::kWgcCapturerWin:
      sequential_id = SequentialDesktopCapturerId::kWgcCapturerWin;
      break;
    case DesktopCapturerId::kWindowCapturerWinGdi:
      sequential_id = SequentialDesktopCapturerId::kWindowCapturerWinGdi;
      break;
    case DesktopCapturerId::kScreenCapturerWinGdi:
      sequential_id = SequentialDesktopCapturerId::kScreenCapturerWinGdi;
      break;
    case DesktopCapturerId::kScreenCapturerWinDirectx:
      sequential_id = SequentialDesktopCapturerId::kScreenCapturerWinDirectx;
      break;
    case DesktopCapturerId::kUnknown:
    default:
      sequential_id = SequentialDesktopCapturerId::kUnknown;
  }
  RTC_HISTOGRAM_ENUMERATION(
      "WebRTC.DesktopCapture.Win.DesktopCapturerImpl",
      static_cast<int>(sequential_id),
      static_cast<int>(SequentialDesktopCapturerId::kMaxValue));
}

}  // namespace webrtc
