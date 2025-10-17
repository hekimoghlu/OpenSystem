/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 31, 2023.
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
#pragma once

#if HAVE(SCREEN_CAPTURE_KIT)

#include <wtf/SoftLinking.h>

SOFT_LINK_FRAMEWORK_FOR_HEADER(PAL, ScreenCaptureKit);

SOFT_LINK_CLASS_FOR_HEADER_WITH_AVAILABILITY(PAL, SCWindow, API_AVAILABLE(macos(12.3)))
SOFT_LINK_CLASS_FOR_HEADER_WITH_AVAILABILITY(PAL, SCDisplay, API_AVAILABLE(macos(12.3)))
SOFT_LINK_CLASS_FOR_HEADER_WITH_AVAILABILITY(PAL, SCShareableContent, API_AVAILABLE(macos(12.3)))
SOFT_LINK_CLASS_FOR_HEADER_WITH_AVAILABILITY(PAL, SCContentFilter, API_AVAILABLE(macos(12.3)))
SOFT_LINK_CLASS_FOR_HEADER_WITH_AVAILABILITY(PAL, SCStreamConfiguration, API_AVAILABLE(macos(12.3)))
SOFT_LINK_CLASS_FOR_HEADER_WITH_AVAILABILITY(PAL, SCStream, API_AVAILABLE(macos(12.3)))
SOFT_LINK_CLASS_FOR_HEADER(PAL, SCContentSharingSession)

#if HAVE(SC_CONTENT_SHARING_PICKER)
SOFT_LINK_CLASS_FOR_HEADER(PAL, SCContentSharingPicker)
SOFT_LINK_CLASS_FOR_HEADER(PAL, SCContentSharingPickerConfiguration)
#endif

SOFT_LINK_CONSTANT_FOR_HEADER(PAL, ScreenCaptureKit, SCStreamFrameInfoStatus, NSString *)
#define SCStreamFrameInfoStatus PAL::get_ScreenCaptureKit_SCStreamFrameInfoStatus()
SOFT_LINK_CONSTANT_FOR_HEADER(PAL, ScreenCaptureKit, SCStreamFrameInfoScaleFactor, NSString *)
#define SCStreamFrameInfoScaleFactor PAL::get_ScreenCaptureKit_SCStreamFrameInfoScaleFactor()
SOFT_LINK_CONSTANT_FOR_HEADER(PAL, ScreenCaptureKit, SCStreamFrameInfoContentScale, NSString *)
#define SCStreamFrameInfoContentScale PAL::get_ScreenCaptureKit_SCStreamFrameInfoContentScale()
SOFT_LINK_CONSTANT_FOR_HEADER(PAL, ScreenCaptureKit, SCStreamFrameInfoContentRect, NSString *)
#define SCStreamFrameInfoContentRect PAL::get_ScreenCaptureKit_SCStreamFrameInfoContentRect()

#if HAVE(SC_CONTENT_SHARING_PICKER)
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_HEADER(PAL, ScreenCaptureKit, SCStreamFrameInfoPresenterOverlayContentRect, NSString *)
#define SCStreamFrameInfoPresenterOverlayContentRect PAL::get_ScreenCaptureKit_SCStreamFrameInfoPresenterOverlayContentRect()
#endif

#endif // HAVE(SCREEN_CAPTURE_KIT)
