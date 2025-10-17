/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 11, 2025.
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

// FIXME: Remove once rdar://131328679 is fixed and distributed.
IGNORE_WARNINGS_BEGIN("#warnings")
#include <AccessibilitySupport.h>
IGNORE_WARNINGS_END

#endif

#if !USE(APPLE_INTERNAL_SDK)
typedef enum {
    AXValueStateInvalid = -2,
    AXValueStateEmpty = -1,
    AXValueStateOff,
    AXValueStateOn
} AXValueState;
#endif

WTF_EXTERN_C_BEGIN

void _AXSSetReduceMotionEnabled(Boolean enabled);
void _AXSSetDarkenSystemColors(Boolean enabled);
Boolean _AXSKeyRepeatEnabled();
Boolean _AXSApplicationAccessibilityEnabled();
void _AXSApplicationAccessibilitySetEnabled(Boolean enabled);
extern CFStringRef kAXSApplicationAccessibilityEnabledNotification;

extern CFStringRef kAXSReduceMotionPreference;

extern CFStringRef kAXSReduceMotionChangedNotification;
extern CFStringRef kAXSIncreaseButtonLegibilityNotification;
extern CFStringRef kAXSEnhanceTextLegibilityChangedNotification;
extern CFStringRef kAXSDarkenSystemColorsEnabledNotification;
extern CFStringRef kAXSInvertColorsEnabledNotification;

AXValueState _AXSReduceMotionEnabledApp(CFStringRef appID);
AXValueState _AXSIncreaseButtonLegibilityApp(CFStringRef appID);
AXValueState _AXSEnhanceTextLegibilityEnabledApp(CFStringRef appID);
AXValueState _AXDarkenSystemColorsApp(CFStringRef appID);
AXValueState _AXSInvertColorsEnabledApp(CFStringRef appID);
Boolean _AXSEnhanceTextLegibilityEnabled();

void _AXSSetReduceMotionEnabledApp(AXValueState enabled, CFStringRef appID);
void _AXSSetIncreaseButtonLegibilityApp(AXValueState enabled, CFStringRef appID);
void _AXSSetEnhanceTextLegibilityEnabledApp(AXValueState enabled, CFStringRef appID);
void _AXSSetDarkenSystemColorsApp(AXValueState enabled, CFStringRef appID);
void _AXSInvertColorsSetEnabledApp(AXValueState enabled, CFStringRef appID);

extern CFStringRef kAXSReduceMotionAutoplayAnimatedImagesChangedNotification;
extern Boolean _AXSReduceMotionAutoplayAnimatedImagesEnabled(void);

extern CFStringRef kAXSPrefersNonBlinkingCursorIndicatorDidChangeNotification;
extern Boolean _AXSPrefersNonBlinkingCursorIndicator(void);

extern CFStringRef kAXSFullKeyboardAccessEnabledNotification;
Boolean _AXSFullKeyboardAccessEnabled();

extern CFStringRef kAXSAccessibilityPreferenceDomain;

WTF_EXTERN_C_END
