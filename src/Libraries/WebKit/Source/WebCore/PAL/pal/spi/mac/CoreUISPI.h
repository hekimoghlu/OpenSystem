/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 3, 2025.
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

#if USE(APPLE_INTERNAL_SDK)

#include <CoreUI/CoreUI.h>

#else

extern const CFStringRef kCUIAnimationStartTimeKey;

extern const CFStringRef kCUIAnimationTimeKey;

extern const CFStringRef kCUIIsFlippedKey;

extern const CFStringRef kCUIMaskOnlyKey;

extern const CFStringRef kCUIOrientationKey;
extern const CFStringRef kCUIOrientHorizontal;

extern const CFStringRef kCUIPresentationStateKey;
extern const CFStringRef kCUIPresentationStateActiveKey;
extern const CFStringRef kCUIPresentationStateInactive;

extern const CFStringRef kCUIScaleKey;

extern const CFStringRef kCUISizeKey;
extern const CFStringRef kCUISizeMini;
extern const CFStringRef kCUISizeSmall;
extern const CFStringRef kCUISizeRegular;

extern const CFStringRef kCUIStateKey;
extern const CFStringRef kCUIStateActive;
extern const CFStringRef kCUIStateDisabled;
extern const CFStringRef kCUIStatePressed;

extern const CFStringRef kCUIUserInterfaceLayoutDirectionKey;
extern const CFStringRef kCUIUserInterfaceLayoutDirectionLeftToRight;
extern const CFStringRef kCUIUserInterfaceLayoutDirectionRightToLeft;

extern const CFStringRef kCUIValueKey;

extern const CFStringRef kCUIWidgetKey;
extern const CFStringRef kCUIWidgetButtonComboBox;
extern const CFStringRef kCUIWidgetButtonLittleArrows;
extern const CFStringRef kCUIWidgetProgressIndeterminateBar;
extern const CFStringRef kCUIWidgetProgressBar;
extern const CFStringRef kCUIWidgetScrollBarTrackCorner;
extern const CFStringRef kCUIWidgetSwitchKnob;
extern const CFStringRef kCUIWidgetSwitchBorder;
extern const CFStringRef kCUIWidgetSwitchFill;
extern const CFStringRef kCUIWidgetSwitchFillMask;
extern const CFStringRef kCUIWidgetSwitchOnOffLabel;

#endif
