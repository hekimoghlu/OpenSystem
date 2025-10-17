/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 31, 2025.
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

#if ENABLE(GAMEPAD) && PLATFORM(COCOA)

#import <CoreHaptics/CoreHaptics.h>
#import <wtf/SoftLinking.h>

SOFT_LINK_FRAMEWORK_FOR_HEADER(WebCore, CoreHaptics)

SOFT_LINK_CLASS_FOR_HEADER(WebCore, CHHapticEngine)
SOFT_LINK_CLASS_FOR_HEADER(WebCore, CHHapticPattern)

SOFT_LINK_CONSTANT_FOR_HEADER(WebCore, CoreHaptics, CHHapticPatternKeyEvent, NSString *);
#define CHHapticPatternKeyEvent WebCore::get_CoreHaptics_CHHapticPatternKeyEvent()
SOFT_LINK_CONSTANT_FOR_HEADER(WebCore, CoreHaptics, CHHapticPatternKeyEventType, NSString *);
#define CHHapticPatternKeyEventType WebCore::get_CoreHaptics_CHHapticPatternKeyEventType()
SOFT_LINK_CONSTANT_FOR_HEADER(WebCore, CoreHaptics, CHHapticEventTypeHapticTransient, NSString *);
#define CHHapticEventTypeHapticTransient WebCore::get_CoreHaptics_CHHapticEventTypeHapticTransient()
SOFT_LINK_CONSTANT_FOR_HEADER(WebCore, CoreHaptics, CHHapticEventTypeHapticContinuous, NSString *);
#define CHHapticEventTypeHapticContinuous WebCore::get_CoreHaptics_CHHapticEventTypeHapticContinuous()
SOFT_LINK_CONSTANT_FOR_HEADER(WebCore, CoreHaptics, CHHapticPatternKeyTime, NSString *);
#define CHHapticPatternKeyTime WebCore::get_CoreHaptics_CHHapticPatternKeyTime()
SOFT_LINK_CONSTANT_FOR_HEADER(WebCore, CoreHaptics, CHHapticPatternKeyEventParameters, NSString *);
#define CHHapticPatternKeyEventParameters WebCore::get_CoreHaptics_CHHapticPatternKeyEventParameters()
SOFT_LINK_CONSTANT_FOR_HEADER(WebCore, CoreHaptics, CHHapticPatternKeyParameterID, NSString *);
#define CHHapticPatternKeyParameterID WebCore::get_CoreHaptics_CHHapticPatternKeyParameterID()
SOFT_LINK_CONSTANT_FOR_HEADER(WebCore, CoreHaptics, CHHapticPatternKeyParameterValue, NSString *);
#define CHHapticPatternKeyParameterValue WebCore::get_CoreHaptics_CHHapticPatternKeyParameterValue()
SOFT_LINK_CONSTANT_FOR_HEADER(WebCore, CoreHaptics, CHHapticEventParameterIDHapticIntensity, NSString *);
#define CHHapticEventParameterIDHapticIntensity WebCore::get_CoreHaptics_CHHapticEventParameterIDHapticIntensity()
SOFT_LINK_CONSTANT_FOR_HEADER(WebCore, CoreHaptics, CHHapticPatternKeyEventDuration, NSString *);
#define CHHapticPatternKeyEventDuration WebCore::get_CoreHaptics_CHHapticPatternKeyEventDuration()
SOFT_LINK_CONSTANT_FOR_HEADER(WebCore, CoreHaptics, CHHapticPatternKeyPattern, NSString *);
#define CHHapticPatternKeyPattern WebCore::get_CoreHaptics_CHHapticPatternKeyPattern()

#endif // ENABLE(GAMEPAD) && PLATFORM(COCOA)
