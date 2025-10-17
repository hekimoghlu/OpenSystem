/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 17, 2023.
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
#ifndef IOHIDUtility_h
#define IOHIDUtility_h

#include <IOKit/IOTypes.h>
#include <IOKit/hid/IOHIDUsageTables.h>

enum {
    kKeyMaskCtrl             = 0x0001,
    kKeyMaskShift            = 0x0002,
    kKeyMaskAlt              = 0x0004,
    kKeyMaskLeftCommand      = 0x0008,
    kKeyMaskRightCommand     = 0x0010,
    kKeyMaskPeriod           = 0x0020,
    kKeyMaskComma            = 0x0040,
    kKeyMaskSlash            = 0x0080,
    kKeyMaskEsc              = 0x0100,
    kKeyMaskFn               = 0x0200,
    kKeyMaskDelete           = 0x0400,
    kKeyMaskPower            = 0x0800,
    kKeyMaskUnknown          = 0x80000000
};

struct Key {
    uint64_t _value;
    bool _modified;
    Key ():_value(0), _modified(false){}
    Key (uint32_t usagePage, uint32_t usage):_value(((uint64_t)usagePage << 32) | usage), _modified(false) {}
    Key (uint32_t usagePage, uint32_t usage, bool modified):_value(((uint64_t)usagePage << 32) | usage), _modified(modified) {}
    Key (uint64_t usageAndUsagePage): _value (usageAndUsagePage), _modified(false) {}
    friend bool operator==(const Key &lhs, const Key &rhs) {
        return (lhs._value == rhs._value);
    }
    friend bool operator<(const Key &lhs, const Key &rhs) {
        return (lhs._value < rhs._value);
    }
    uint32_t usage () const {return ((uint32_t*)&_value)[0];}
    uint32_t usagePage () const {return((uint32_t*)&_value)[1];}
    bool isValid () const {return _value != 0;}
    bool isModifier () const;
    bool isTopRow () const {
        bool result = false;
        if ((usagePage() == kHIDPage_KeyboardOrKeypad) &&
            (((usage() >= kHIDUsage_Keyboard1) && (usage() <= kHIDUsage_Keyboard0)) ||
              (usage() == kHIDUsage_KeyboardHyphen) ||
              (usage() == kHIDUsage_KeyboardEqualSign) ||
              (usage() == kHIDUsage_KeyboardGraveAccentAndTilde) ||
              (usage() == kHIDUsage_KeyboardDeleteOrBackspace))) {
            result = true;
        }
        return result;
    }
    uint32_t modifierMask () const;

};

struct KeyAttribute {
    uint32_t  _flags;
    KeyAttribute (uint32_t  flags = 0):_flags(flags) {};
};

#endif /* IOHIDUtility_h */
