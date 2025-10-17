/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 21, 2022.
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
#ifndef MODULES_DESKTOP_CAPTURE_MAC_DESKTOP_FRAME_UTILS_H_
#define MODULES_DESKTOP_CAPTURE_MAC_DESKTOP_FRAME_UTILS_H_

#include <memory>

#include "modules/desktop_capture/desktop_frame.h"
#include "modules/desktop_capture/mac/desktop_frame_cgimage.h"
#include "rtc_base/system/rtc_export.h"

namespace webrtc {

std::unique_ptr<DesktopFrame> RTC_EXPORT
CreateDesktopFrameFromCGImage(rtc::ScopedCFTypeRef<CGImageRef> cg_image);

}  // namespace webrtc

#endif  // MODULES_DESKTOP_CAPTURE_MAC_DESKTOP_FRAME_UTILS_H_
