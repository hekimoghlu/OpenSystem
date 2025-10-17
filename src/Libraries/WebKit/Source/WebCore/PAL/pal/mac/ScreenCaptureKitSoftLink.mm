/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 6, 2023.
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
#include "config.h"

#if HAVE(SCREEN_CAPTURE_KIT)

#import <dispatch/dispatch.h>
#import <wtf/SoftLinking.h>

SOFT_LINK_FRAMEWORK_FOR_SOURCE_WITH_EXPORT(PAL, ScreenCaptureKit, PAL_EXPORT)

SOFT_LINK_CLASS_FOR_SOURCE_OPTIONAL_WITH_EXPORT_AND_AVAILABILITY(PAL, ScreenCaptureKit, SCWindow, PAL_EXPORT, API_AVAILABLE(macos(12.3)))
SOFT_LINK_CLASS_FOR_SOURCE_OPTIONAL_WITH_EXPORT_AND_AVAILABILITY(PAL, ScreenCaptureKit, SCDisplay, PAL_EXPORT, API_AVAILABLE(macos(12.3)))
SOFT_LINK_CLASS_FOR_SOURCE_OPTIONAL_WITH_EXPORT_AND_AVAILABILITY(PAL, ScreenCaptureKit, SCShareableContent, PAL_EXPORT, API_AVAILABLE(macos(12.3)))
SOFT_LINK_CLASS_FOR_SOURCE_OPTIONAL_WITH_EXPORT_AND_AVAILABILITY(PAL, ScreenCaptureKit, SCContentFilter, PAL_EXPORT, API_AVAILABLE(macos(12.3)))
SOFT_LINK_CLASS_FOR_SOURCE_OPTIONAL_WITH_EXPORT_AND_AVAILABILITY(PAL, ScreenCaptureKit, SCStreamConfiguration, PAL_EXPORT, API_AVAILABLE(macos(12.3)))
SOFT_LINK_CLASS_FOR_SOURCE_OPTIONAL_WITH_EXPORT_AND_AVAILABILITY(PAL, ScreenCaptureKit, SCStream, PAL_EXPORT, API_AVAILABLE(macos(12.3)))
SOFT_LINK_CLASS_FOR_SOURCE_OPTIONAL_WITH_EXPORT(PAL, ScreenCaptureKit, SCContentSharingSession, PAL_EXPORT)

#if HAVE(SC_CONTENT_SHARING_PICKER)
SOFT_LINK_CLASS_FOR_SOURCE_OPTIONAL_WITH_EXPORT(PAL, ScreenCaptureKit, SCContentSharingPicker, PAL_EXPORT)
SOFT_LINK_CLASS_FOR_SOURCE_OPTIONAL_WITH_EXPORT(PAL, ScreenCaptureKit, SCContentSharingPickerConfiguration, PAL_EXPORT)
#endif

SOFT_LINK_CONSTANT_FOR_SOURCE_WITH_EXPORT(PAL, ScreenCaptureKit, SCStreamFrameInfoStatus, NSString *, PAL_EXPORT)
SOFT_LINK_CONSTANT_FOR_SOURCE_WITH_EXPORT(PAL, ScreenCaptureKit, SCStreamFrameInfoScaleFactor, NSString *, PAL_EXPORT)
SOFT_LINK_CONSTANT_FOR_SOURCE_WITH_EXPORT(PAL, ScreenCaptureKit, SCStreamFrameInfoContentScale, NSString *, PAL_EXPORT)
SOFT_LINK_CONSTANT_FOR_SOURCE_WITH_EXPORT(PAL, ScreenCaptureKit, SCStreamFrameInfoContentRect, NSString *, PAL_EXPORT)
SOFT_LINK_CONSTANT_FOR_SOURCE_WITH_EXPORT(PAL, ScreenCaptureKit, SCStreamFrameInfoBoundingRect, NSString *, PAL_EXPORT)

#if HAVE(SC_CONTENT_SHARING_PICKER)
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_SOURCE_WITH_EXPORT(PAL, ScreenCaptureKit, SCStreamFrameInfoPresenterOverlayContentRect, NSString *, PAL_EXPORT)
#endif

#endif // PLATFORM(MAC)
