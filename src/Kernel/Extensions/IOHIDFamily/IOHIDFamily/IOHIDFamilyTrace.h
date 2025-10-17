/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 13, 2021.
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
#ifndef _IOKIT_HID_IOHIDFAMILYTRACE_H // {
#define _IOKIT_HID_IOHIDFAMILYTRACE_H

#include <sys/kdebug.h>

#define IOHID_DEBUG_CODE(code)          IOKDBG_CODE(DBG_IOHID, code)
#define IOHID_DEBUG(code, a, b, c, d)   KERNEL_DEBUG_CONSTANT(IOHID_DEBUG_CODE(code), a, b, c, d, 0)

#define IOHIDTraceStart(code, a, b, c, d)     KERNEL_DEBUG_CONSTANT(IOKDBG_CODE(DBG_IOHID, code) | DBG_FUNC_START, a, b, c, d, 0)
#define IOHIDTraceEnd(code, a, b, c, d)       KERNEL_DEBUG_CONSTANT(IOKDBG_CODE(DBG_IOHID, code) | DBG_FUNC_END, a, b, c, d, 0)

enum kIOHIDDebugCodes {
    kIOHIDDebugCode_Unexpected,                 // 0  0x5230000
    kIOHIDDebugCode_KeyboardLEDThreadTrigger,
    kIOHIDDebugCode_KeyboardLEDThreadActive,
    kIOHIDDebugCode_KeyboardSetParam,
    kIOHIDDebugCode_KeyboardCapsThreadTrigger,  // 4  0x5230010
    kIOHIDDebugCode_KeyboardCapsThreadActive,
    kIOHIDDebugCode_PostEvent,
    kIOHIDDebugCode_NewUserClient,
    kIOHIDDebugCode_InturruptReport,            // 8  0x5230020
    kIOHIDDebugCode_DispatchScroll,
    kIOHIDDebugCode_DispatchRelativePointer,
    kIOHIDDebugCode_DispatchAbsolutePointer,
    kIOHIDDebugCode_DispatchKeyboard,           // 12 0x5230030
    kIOHIDDebugCode_EjectCallback,
    kIOHIDDebugCode_CapsCallback,
    kIOHIDDebugCode_HandleReport,
    kIOHIDDebugCode_DispatchTabletPointer,      // 16 0x5230040
    kIOHIDDebugCode_DispatchTabletProx,
    kIOHIDDebugCode_DispatchHIDEvent,
    kIOHIDDebugCode_CalculatedCapsDelay,
    kIOHIDDebugCode_ExtPostEvent,               // 20 0x5230050
    kIOHIDDebugCode_RelativePointerEventTiming,
    kIOHIDDebugCode_RelativePointerEventScaling,
    kIOHIDDebugCode_Profiling,
    kIOHIDDebugCode_DisplayTickle,              // 24 0x5230060
    kIOHIDDebugCode_ExtSetLocation,
    kIOHIDDebugCode_SetCursorPosition,
    kIOHIDDebugCode_PowerStateChangeEvent,
    kIOHIDDebugCode_DispatchDigitizer,          // 28 0x5230070
    kIOHIDDebugCode_Scheduling,
    kIOHIDDebugCode_HIDUserDeviceEnqueueFail,
    kIOHIDDebugCode_HIDDeviceEnqueueFail,
    kIOHIDDebugCode_HIDEventServiceEnqueueFail, // 32 0x5230080
    kIOHIDDebugCode_DK_Intf_HandleReport,
    kIOHIDDebugCode_Invalid
};

#endif // _IOKIT_HID_IOHIDFAMILYTRACE_H }
