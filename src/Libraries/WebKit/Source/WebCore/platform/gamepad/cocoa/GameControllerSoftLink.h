/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 2, 2024.
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

#import "GameControllerSPI.h"
#import <wtf/SoftLinking.h>

SOFT_LINK_FRAMEWORK_FOR_HEADER(WebCore, GameController)
SOFT_LINK_CLASS_FOR_HEADER(WebCore, GCController)
SOFT_LINK_CLASS_FOR_HEADER(WebCore, GCControllerButtonInput)

SOFT_LINK_CONSTANT_MAY_FAIL_FOR_HEADER(WebCore, GameController, GCInputButtonA, NSString *)
#define GCInputButtonA WebCore::get_GameController_GCInputButtonA()
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_HEADER(WebCore, GameController, GCInputButtonB, NSString *)
#define GCInputButtonB WebCore::get_GameController_GCInputButtonB()
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_HEADER(WebCore, GameController, GCInputButtonX, NSString *)
#define GCInputButtonX WebCore::get_GameController_GCInputButtonX()
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_HEADER(WebCore, GameController, GCInputButtonY, NSString *)
#define GCInputButtonY WebCore::get_GameController_GCInputButtonY()
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_HEADER(WebCore, GameController, GCInputButtonHome, NSString *)
#define GCInputButtonHome WebCore::get_GameController_GCInputButtonHome()
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_HEADER(WebCore, GameController, GCInputButtonMenu, NSString *)
#define GCInputButtonMenu WebCore::get_GameController_GCInputButtonMenu()
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_HEADER(WebCore, GameController, GCInputButtonOptions, NSString *)
#define GCInputButtonOptions WebCore::get_GameController_GCInputButtonOptions()
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_HEADER(WebCore, GameController, GCInputDirectionPad, NSString *)
#define GCInputDirectionPad WebCore::get_GameController_GCInputDirectionPad()
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_HEADER(WebCore, GameController, GCInputLeftShoulder, NSString *)
#define GCInputLeftShoulder WebCore::get_GameController_GCInputLeftShoulder()
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_HEADER(WebCore, GameController, GCInputLeftTrigger, NSString *)
#define GCInputLeftTrigger WebCore::get_GameController_GCInputLeftTrigger()
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_HEADER(WebCore, GameController, GCInputLeftThumbstick, NSString *)
#define GCInputLeftThumbstick WebCore::get_GameController_GCInputLeftThumbstick()
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_HEADER(WebCore, GameController, GCInputLeftThumbstickButton, NSString *)
#define GCInputLeftThumbstickButton WebCore::get_GameController_GCInputLeftThumbstickButton()
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_HEADER(WebCore, GameController, GCInputRightShoulder, NSString *)
#define GCInputRightShoulder WebCore::get_GameController_GCInputRightShoulder()
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_HEADER(WebCore, GameController, GCInputRightTrigger, NSString *)
#define GCInputRightTrigger WebCore::get_GameController_GCInputRightTrigger()
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_HEADER(WebCore, GameController, GCInputRightThumbstick, NSString *)
#define GCInputRightThumbstick WebCore::get_GameController_GCInputRightThumbstick()
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_HEADER(WebCore, GameController, GCInputRightThumbstickButton, NSString *)
#define GCInputRightThumbstickButton WebCore::get_GameController_GCInputRightThumbstickButton()

SOFT_LINK_CONSTANT_MAY_FAIL_FOR_HEADER(WebCore, GameController, GCControllerDidConnectNotification, NSString *)
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_HEADER(WebCore, GameController, GCControllerDidDisconnectNotification, NSString *)

SOFT_LINK_CONSTANT_MAY_FAIL_FOR_HEADER(WebCore, GameController, GCHapticsLocalityLeftHandle, NSString *)
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_HEADER(WebCore, GameController, GCHapticsLocalityRightHandle, NSString *)
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_HEADER(WebCore, GameController, GCHapticsLocalityLeftTrigger, NSString *)
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_HEADER(WebCore, GameController, GCHapticsLocalityRightTrigger, NSString *)

#if HAVE(MULTIGAMEPADPROVIDER_SUPPORT)
SOFT_LINK_FUNCTION_FOR_HEADER(WebCore, GameController, ControllerClassForService, Class, (IOHIDServiceClientRef service), (service))
#endif

#if PLATFORM(VISION) && __has_include(<GameController/GCEventInteraction.h>)
#import <GameController/GCEventInteraction.h>

SOFT_LINK_CLASS_FOR_HEADER(WebCore, GCEventInteraction)
#endif

#if USE(APPLE_INTERNAL_SDK) && __has_include(<WebKitAdditions/GameControllerSoftLinkAdditions.h>)
#import <WebKitAdditions/GameControllerSoftLinkAdditions.h>
#endif

#endif // ENABLE(GAMEPAD) && PLATFORM(COCOA)
