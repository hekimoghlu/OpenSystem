/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 13, 2023.
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

#if ENABLE(GAMEPAD) && PLATFORM(MAC)

#import <IOKit/hid/IOHIDUsageTables.h>

namespace WebCore {

constexpr const uint64_t hidHatswitchFullUsage = ((uint64_t)kHIDPage_GenericDesktop) << 32 | kHIDUsage_GD_Hatswitch;
constexpr const uint64_t hidPointerFullUsage = ((uint64_t)kHIDPage_GenericDesktop) << 32 | kHIDUsage_GD_Pointer;
constexpr const uint64_t hidXAxisFullUsage = ((uint64_t)kHIDPage_GenericDesktop) << 32 | kHIDUsage_GD_X;
constexpr const uint64_t hidYAxisFullUsage = ((uint64_t)kHIDPage_GenericDesktop) << 32 | kHIDUsage_GD_Y;
constexpr const uint64_t hidZAxisFullUsage = ((uint64_t)kHIDPage_GenericDesktop) << 32 | kHIDUsage_GD_Z;
constexpr const uint64_t hidRzAxisFullUsage = ((uint64_t)kHIDPage_GenericDesktop) << 32 | kHIDUsage_GD_Rz;

constexpr const uint64_t hidAcceleratorFullUsage = ((uint64_t)kHIDPage_Simulation) << 32 | kHIDUsage_Sim_Accelerator;
constexpr const uint64_t hidBrakeFullUsage = ((uint64_t)kHIDPage_Simulation) << 32 | kHIDUsage_Sim_Brake;

constexpr const uint64_t hidButton1FullUsage = ((uint64_t)kHIDPage_Button) << 32 | kHIDUsage_Button_1;
constexpr const uint64_t hidButton2FullUsage = ((uint64_t)kHIDPage_Button) << 32 | (kHIDUsage_Button_1 + 1);
constexpr const uint64_t hidButton3FullUsage = ((uint64_t)kHIDPage_Button) << 32 | (kHIDUsage_Button_1 + 2);
constexpr const uint64_t hidButton4FullUsage = ((uint64_t)kHIDPage_Button) << 32 | (kHIDUsage_Button_1 + 3);
constexpr const uint64_t hidButton5FullUsage = ((uint64_t)kHIDPage_Button) << 32 | (kHIDUsage_Button_1 + 4);
constexpr const uint64_t hidButton6FullUsage = ((uint64_t)kHIDPage_Button) << 32 | (kHIDUsage_Button_1 + 5);
constexpr const uint64_t hidButton7FullUsage = ((uint64_t)kHIDPage_Button) << 32 | (kHIDUsage_Button_1 + 6);
constexpr const uint64_t hidButton8FullUsage = ((uint64_t)kHIDPage_Button) << 32 | (kHIDUsage_Button_1 + 7);
constexpr const uint64_t hidButton9FullUsage = ((uint64_t)kHIDPage_Button) << 32 | (kHIDUsage_Button_1 + 8);
constexpr const uint64_t hidButton10FullUsage = ((uint64_t)kHIDPage_Button) << 32 | (kHIDUsage_Button_1 + 9);
constexpr const uint64_t hidButton11FullUsage = ((uint64_t)kHIDPage_Button) << 32 | (kHIDUsage_Button_1 + 10);
constexpr const uint64_t hidButton12FullUsage = ((uint64_t)kHIDPage_Button) << 32 | (kHIDUsage_Button_1 + 11);
constexpr const uint64_t hidButton13FullUsage = ((uint64_t)kHIDPage_Button) << 32 | (kHIDUsage_Button_1 + 12);
constexpr const uint64_t hidButton14FullUsage = ((uint64_t)kHIDPage_Button) << 32 | (kHIDUsage_Button_1 + 13);
constexpr const uint64_t hidButton15FullUsage = ((uint64_t)kHIDPage_Button) << 32 | (kHIDUsage_Button_1 + 14);
constexpr const uint64_t hidButton17FullUsage = ((uint64_t)kHIDPage_Button) << 32 | (kHIDUsage_Button_1 + 16);
constexpr const uint64_t hidButton18FullUsage = ((uint64_t)kHIDPage_Button) << 32 | (kHIDUsage_Button_1 + 17);
constexpr const uint64_t hidButton19FullUsage = ((uint64_t)kHIDPage_Button) << 32 | (kHIDUsage_Button_1 + 18);
constexpr const uint64_t hidButton20FullUsage = ((uint64_t)kHIDPage_Button) << 32 | (kHIDUsage_Button_1 + 19);

} // namespace WebCore

#endif // ENABLE(GAMEPAD) && PLATFORM(MAC)
