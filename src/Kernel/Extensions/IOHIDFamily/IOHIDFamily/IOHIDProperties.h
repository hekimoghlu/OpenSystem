/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 28, 2024.
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
#ifndef IOHIDProperties_h
#define IOHIDProperties_h

#include <IOKit/hid/IOHIDEventServiceKeys.h>

/*!
 * @define      kIOHIDMouseAccelerationType
 *
 * @abstract    CFNumber that contains the mouse acceleration value.
 */
#define kIOHIDMouseAccelerationType                 "HIDMouseAcceleration"

/*!
 * @define      kIOHIDPointerButtonMode
 *
 * @abstract    CFNumber containing the current pointer button mode.
 *              See IOHIDButtonModes enumerator for possible modes.
 */
#define kIOHIDPointerButtonMode                     "HIDPointerButtonMode"
#define kIOHIDPointerButtonModeKey                  kIOHIDPointerButtonMode

/*!
 * @define      kIOHIDUserUsageMapKey
 *
 * @abstract    CFArray of dictionaries that contain user defined key mappings.
 */
#define kIOHIDUserKeyUsageMapKey                     "UserKeyMapping"

/*!
 * @define      kIOHIDKeyboardCapsLockDelayOverride
 *
 * @abstract    CFNumber containing the delay (in ms) before the caps lock key is activated.
 */
#define kIOHIDKeyboardCapsLockDelayOverride         "CapsLockDelayOverride"
#define kIOHIDKeyboardCapsLockDelayOverrideKey      kIOHIDKeyboardCapsLockDelayOverride

/*!
 * @define      kIOHIDServiceEjectDelayKey
 *
 * @abstract    CFNumber containing the delay (in ms) before the eject key is activated.
 */
#define kIOHIDServiceEjectDelayKey                  "EjectDelay"

/*!
 * @define      kIOHIDServiceLockKeyDelayKey
 *
 * @abstract    CFNumber containing the delay (in ms) before the lock key is activated.
 */
#define kIOHIDServiceLockKeyDelayKey				 "LockKeyDelay"

/*!
 * @define      kIOHIDServiceInitialKeyRepeatDelayKey
 *
 * @abstract    CFNumber containing the delay (in ns) before the initial key repeat.
 *              If value is 0, there are no repeats.
 */
#define kIOHIDServiceInitialKeyRepeatDelayKey       "HIDInitialKeyRepeat"

/*!
 * @define      kIOHIDServiceKeyRepeatDelayKey
 *
 * @abstract    CFNumber containing the delay (in ns) for subsequent key repeats.
 *              If value is 0, there are no repeats (including initial).
 */
#define kIOHIDServiceKeyRepeatDelayKey              "HIDKeyRepeat"

/*!
 * @define      kIOHIDIdleTimeMicrosecondsKey
 *
 * @abstract    CFNumber containing the HID idle time in microseconds.
 */
#define kIOHIDIdleTimeMicrosecondsKey               "HIDIdleTimeMicroseconds"

/*!
 * @define      kIOHIDServiceCapsLockStateKey
 *
 * @abstract    CFBoolean for setting/getting the caps lock state of the
 *              service. The caps lock LED will be updated to reflect the state.
 */
#define kIOHIDServiceCapsLockStateKey                "HIDCapsLockState"

#endif /* IOHIDProperties_h */
