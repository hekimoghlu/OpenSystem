/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 27, 2022.
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

#import "GameControllerSPI.h"
#import <wtf/SoftLinking.h>

SOFT_LINK_FRAMEWORK_FOR_SOURCE(WebCore, GameController)
SOFT_LINK_CLASS_FOR_SOURCE_OPTIONAL_WITH_EXPORT(WebCore, GameController, GCController, WEBCORE_EXPORT)
SOFT_LINK_CLASS_FOR_SOURCE_OPTIONAL_WITH_EXPORT(WebCore, GameController, GCControllerButtonInput, WEBCORE_EXPORT)

SOFT_LINK_CONSTANT_MAY_FAIL_FOR_SOURCE_WITH_EXPORT(WebCore, GameController, GCInputButtonA, NSString *, WEBCORE_EXPORT)
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_SOURCE_WITH_EXPORT(WebCore, GameController, GCInputButtonB, NSString *, WEBCORE_EXPORT)
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_SOURCE_WITH_EXPORT(WebCore, GameController, GCInputButtonX, NSString *, WEBCORE_EXPORT)
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_SOURCE_WITH_EXPORT(WebCore, GameController, GCInputButtonY, NSString *, WEBCORE_EXPORT)
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_SOURCE_WITH_EXPORT(WebCore, GameController, GCInputButtonHome, NSString *, WEBCORE_EXPORT)
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_SOURCE_WITH_EXPORT(WebCore, GameController, GCInputButtonMenu, NSString *, WEBCORE_EXPORT)
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_SOURCE_WITH_EXPORT(WebCore, GameController, GCInputButtonOptions, NSString *, WEBCORE_EXPORT)
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_SOURCE_WITH_EXPORT(WebCore, GameController, GCInputDirectionPad, NSString *, WEBCORE_EXPORT)
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_SOURCE_WITH_EXPORT(WebCore, GameController, GCInputLeftShoulder, NSString *, WEBCORE_EXPORT)
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_SOURCE_WITH_EXPORT(WebCore, GameController, GCInputLeftTrigger, NSString *, WEBCORE_EXPORT)
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_SOURCE_WITH_EXPORT(WebCore, GameController, GCInputLeftThumbstick, NSString *, WEBCORE_EXPORT)
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_SOURCE_WITH_EXPORT(WebCore, GameController, GCInputLeftThumbstickButton, NSString *, WEBCORE_EXPORT)
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_SOURCE_WITH_EXPORT(WebCore, GameController, GCInputRightShoulder, NSString *, WEBCORE_EXPORT)
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_SOURCE_WITH_EXPORT(WebCore, GameController, GCInputRightTrigger, NSString *, WEBCORE_EXPORT)
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_SOURCE_WITH_EXPORT(WebCore, GameController, GCInputRightThumbstick, NSString *, WEBCORE_EXPORT)
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_SOURCE_WITH_EXPORT(WebCore, GameController, GCInputRightThumbstickButton, NSString *, WEBCORE_EXPORT)

SOFT_LINK_CONSTANT_MAY_FAIL_FOR_SOURCE_WITH_EXPORT(WebCore, GameController, GCControllerDidConnectNotification, NSString *, WEBCORE_EXPORT)
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_SOURCE_WITH_EXPORT(WebCore, GameController, GCControllerDidDisconnectNotification, NSString *, WEBCORE_EXPORT)

SOFT_LINK_CONSTANT_MAY_FAIL_FOR_SOURCE_WITH_EXPORT(WebCore, GameController, GCHapticsLocalityLeftHandle, NSString *, WEBCORE_EXPORT)
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_SOURCE_WITH_EXPORT(WebCore, GameController, GCHapticsLocalityRightHandle, NSString *, WEBCORE_EXPORT)
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_SOURCE_WITH_EXPORT(WebCore, GameController, GCHapticsLocalityLeftTrigger, NSString *, WEBCORE_EXPORT)
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_SOURCE_WITH_EXPORT(WebCore, GameController, GCHapticsLocalityRightTrigger, NSString *, WEBCORE_EXPORT)

#if HAVE(MULTIGAMEPADPROVIDER_SUPPORT)
SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, GameController, ControllerClassForService, Class, (IOHIDServiceClientRef service), (service))
#endif

#if PLATFORM(VISION) && __has_include(<GameController/GCEventInteraction.h>)
SOFT_LINK_CLASS_FOR_SOURCE_OPTIONAL_WITH_EXPORT(WebCore, GameController, GCEventInteraction, WEBCORE_EXPORT)
#endif

#if USE(APPLE_INTERNAL_SDK) && __has_include(<WebKitAdditions/GameControllerSoftLinkAdditions.mm>)
#import <WebKitAdditions/GameControllerSoftLinkAdditions.mm>
#endif

#endif // ENABLE(GAMEPAD) && PLATFORM(COCOA)
