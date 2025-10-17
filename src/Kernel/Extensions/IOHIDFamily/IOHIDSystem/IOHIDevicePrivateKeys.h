/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 14, 2023.
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
#ifndef _IOHIDEVICE_PRIVATE_KEYS_H
#define _IOHIDEVICE_PRIVATE_KEYS_H

enum {
    kAccelMouse                 = 0x0001,
    kAccelScroll                = 0x0002,
    kAbsoluteConvertMouse       = 0x0004,
    kAccelScrollConvertPointer  = 0x0008,
    kAccelNoScrollAcceleration  = 0x0010
};

enum {
    kScrollTypeContinuous       = 0x0001,
    kScrollTypeZoom             = 0x0002,
    kScrollTypeMomentumContinue = 0x0004,
    kScrollTypeTouch            = 0x0008,
    kScrollTypeMomentumStart    = 0x0010,
    kScrollTypeMomentumEnd      = 0x0020,    
    kScrollTypeMomentumAny      = kScrollTypeMomentumContinue | kScrollTypeMomentumStart | kScrollTypeMomentumEnd,
    
    kScrollTypeOptionPhaseAny           = 0xff00,
    kScrollTypeOptionPhaseBegan         = 0x0100,
    kScrollTypeOptionPhaseChanged       = 0x0200,
    kScrollTypeOptionPhaseEnded         = 0x0400,
    kScrollTypeOptionPhaseCanceled      = 0x0800,    
    kScrollTypeOptionPhaseMayBegin      = 0x8000,    
};

#define kIOHIDEventServicePropertiesKey "HIDEventServiceProperties"
#define kIOHIDTemporaryParametersKey    "HIDTemporaryParameters"
#define kIOHIDDefaultParametersKey      "HIDDefaultParameters"
#define kIOHIDDeviceParametersKey       "HIDDeviceParameters"
#define kIOHIDDeviceEventIDKey			"HIDDeviceEventID"
#define kIOHIDDeviceScrollWithTrackpadKey "TrackpadScroll" // really should be "HIDDeviceScrollWithTrackpad"
#define kIOHIDDeviceScrollDisableKey    "HIDDeviceScrollDisable"

#endif /* !_IOHIDEVICE_PRIVATE_KEYS_H */

