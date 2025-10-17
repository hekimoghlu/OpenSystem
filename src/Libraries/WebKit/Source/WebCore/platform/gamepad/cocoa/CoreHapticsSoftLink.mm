/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 25, 2023.
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
#import "config.h"

#if ENABLE(GAMEPAD) && PLATFORM(COCOA)
#import <wtf/SoftLinking.h>

SOFT_LINK_FRAMEWORK_FOR_SOURCE(WebCore, CoreHaptics)

SOFT_LINK_CLASS_FOR_SOURCE(WebCore, CoreHaptics, CHHapticEngine)
SOFT_LINK_CLASS_FOR_SOURCE(WebCore, CoreHaptics, CHHapticPattern)

SOFT_LINK_CONSTANT_FOR_SOURCE(WebCore, CoreHaptics, CHHapticPatternKeyEvent, NSString *)
SOFT_LINK_CONSTANT_FOR_SOURCE(WebCore, CoreHaptics, CHHapticPatternKeyEventType, NSString *)
SOFT_LINK_CONSTANT_FOR_SOURCE(WebCore, CoreHaptics, CHHapticEventTypeHapticTransient, NSString *)
SOFT_LINK_CONSTANT_FOR_SOURCE(WebCore, CoreHaptics, CHHapticEventTypeHapticContinuous, NSString *)
SOFT_LINK_CONSTANT_FOR_SOURCE(WebCore, CoreHaptics, CHHapticPatternKeyTime, NSString *)
SOFT_LINK_CONSTANT_FOR_SOURCE(WebCore, CoreHaptics, CHHapticPatternKeyEventParameters, NSString *)
SOFT_LINK_CONSTANT_FOR_SOURCE(WebCore, CoreHaptics, CHHapticEventParameterIDHapticIntensity, NSString *)
SOFT_LINK_CONSTANT_FOR_SOURCE(WebCore, CoreHaptics, CHHapticPatternKeyEventDuration, NSString *)
SOFT_LINK_CONSTANT_FOR_SOURCE(WebCore, CoreHaptics, CHHapticPatternKeyPattern, NSString *)
SOFT_LINK_CONSTANT_FOR_SOURCE(WebCore, CoreHaptics, CHHapticPatternKeyParameterID, NSString *)
SOFT_LINK_CONSTANT_FOR_SOURCE(WebCore, CoreHaptics, CHHapticPatternKeyParameterValue, NSString *)

#endif // ENABLE(GAMEPAD) && PLATFORM(COCOA)
