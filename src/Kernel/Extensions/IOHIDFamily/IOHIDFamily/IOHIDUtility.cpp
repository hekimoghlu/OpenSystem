/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 23, 2022.
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
#include "IOHIDUtility.h"
#include <IOKit/hid/AppleHIDUsageTables.h>

bool Key::isModifier() const {
    bool result = false;
    if ((usagePage() == kHIDPage_KeyboardOrKeypad) &&
        (((usage() >= kHIDUsage_KeyboardLeftControl) && (usage() <= kHIDUsage_KeyboardRightGUI)) || (usage() == kHIDUsage_KeyboardCapsLock))) {
        result = true;
    } else if (((usagePage() == kHIDPage_AppleVendorTopCase) && (usage() == kHIDUsage_AV_TopCase_KeyboardFn)) ||
    			((usagePage() == kHIDPage_AppleVendorKeyboard) && (usage() == kHIDUsage_AppleVendorKeyboard_Function))) {
        result = true;
    }
    return result;
};

uint32_t Key::modifierMask() const {
	if (!isModifier()) {
		return 0;
	}

	switch(usagePage()) {
		case kHIDPage_KeyboardOrKeypad:
			switch(usage()) {
				case kHIDUsage_KeyboardLeftControl:
				case kHIDUsage_KeyboardRightControl:
					return kKeyMaskCtrl;
				case kHIDUsage_KeyboardLeftAlt:
				case kHIDUsage_KeyboardRightAlt:
					return kKeyMaskAlt;
				case kHIDUsage_KeyboardLeftGUI:
					return kKeyMaskLeftCommand;
				case kHIDUsage_KeyboardRightGUI:
					return kKeyMaskRightCommand;
				case kHIDUsage_KeyboardLeftShift:
				case kHIDUsage_KeyboardRightShift:
					return kKeyMaskShift;
				default:
					return 0;
			};
		case kHIDPage_AppleVendorTopCase:
			if (usage() == kHIDUsage_AV_TopCase_KeyboardFn) {
				return kKeyMaskFn;
			} else {
				return 0;
			}
		case kHIDPage_AppleVendorKeyboard:
			if (usage() == kHIDUsage_AppleVendorKeyboard_Function) {
				return kKeyMaskFn;
			} else {
				return 0;
			}
		default:
			return 0;
	};

}
