/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 14, 2024.
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

#if USE(CF)

#import <CoreFoundation/CoreFoundation.h>
#import <wtf/RetainPtr.h>
#import <wtf/cocoa/TypeCastsCocoa.h>

namespace WebKit {

class CoreIPCNumber {
public:
    typedef std::variant<
        char,
        unsigned char,
        short,
        unsigned short,
        int,
        unsigned,
        long,
        unsigned long,
        long long,
        unsigned long long,
        float,
        double
    > NumberHolder;

    static NumberHolder numberHolderForNumber(CFNumberRef number)
    {
        CFNumberType numberType = CFNumberGetType(number);
        bool isNegative = [bridge_cast(number) compare:@(0)] == NSOrderedAscending;

        switch (numberType) {
        case kCFNumberSInt8Type:
            return [bridge_cast(number) charValue ];
        case kCFNumberSInt16Type:
            return [bridge_cast(number) shortValue ];
        case kCFNumberSInt32Type:
            return [bridge_cast(number) intValue ];
        case kCFNumberSInt64Type:
            if (isNegative)
                return [bridge_cast(number) longLongValue ];
            return [bridge_cast(number) unsignedLongLongValue ];
        case kCFNumberFloat32Type:
            return [bridge_cast(number) floatValue ];
        case kCFNumberFloat64Type:
            return [bridge_cast(number) doubleValue ];
        case kCFNumberCharType:
            if (isNegative)
                return [bridge_cast(number) charValue ];
            return [bridge_cast(number) unsignedCharValue ];
        case kCFNumberShortType:
            if (isNegative)
                return [bridge_cast(number) shortValue ];
            return [bridge_cast(number) unsignedShortValue ];
        case kCFNumberIntType:
            if (isNegative)
                return [bridge_cast(number) intValue ];
            return [bridge_cast(number) unsignedIntValue ];
        case kCFNumberLongType:
            if (isNegative)
                return [bridge_cast(number) longValue ];
            return [bridge_cast(number) unsignedLongValue ];
        case kCFNumberLongLongType:
            if (isNegative)
                return [bridge_cast(number) longLongValue ];
            return [bridge_cast(number) unsignedLongLongValue ];
        case kCFNumberFloatType:
            return [bridge_cast(number) floatValue ];
        case kCFNumberDoubleType:
            return [bridge_cast(number) doubleValue ];
        case kCFNumberCFIndexType:
            return [bridge_cast(number) longValue ];
        case kCFNumberNSIntegerType:
            return [bridge_cast(number) longValue ];
        case kCFNumberCGFloatType:
            return [bridge_cast(number) doubleValue ];
        }
        RELEASE_ASSERT_NOT_REACHED();
    }

    CoreIPCNumber(NSNumber *number)
        : CoreIPCNumber(bridge_cast(number))
    {
    }

    CoreIPCNumber(CFNumberRef number)
        : m_numberHolder(numberHolderForNumber(number))
    {
    }

    CoreIPCNumber(NumberHolder numberHolder)
        : m_numberHolder(numberHolder)
    {
    }

    CoreIPCNumber(const CoreIPCNumber& other) = default;
    CoreIPCNumber& operator=(const CoreIPCNumber& other) = default;

    RetainPtr<CFNumberRef> createCFNumber() const
    {
        return WTF::switchOn(m_numberHolder,
            [&] (const char& n) {
                return bridge_cast(adoptNS([[NSNumber alloc] initWithChar: n]));
            },
            [&] (const unsigned char& n) {
                return bridge_cast(adoptNS([[NSNumber alloc] initWithUnsignedChar: n]));
            },
            [&] (const short& n) {
                return bridge_cast(adoptNS([[NSNumber alloc] initWithShort: n]));
            },
            [&] (const unsigned short& n) {
                return bridge_cast(adoptNS([[NSNumber alloc] initWithUnsignedShort: n]));
            },
            [&] (const int& n) {
                return bridge_cast(adoptNS([[NSNumber alloc] initWithInt: n]));
            },
            [&] (const unsigned& n) {
                return bridge_cast(adoptNS([[NSNumber alloc] initWithUnsignedInt: n]));
            },
            [&] (const long& n) {
                return bridge_cast(adoptNS([[NSNumber alloc] initWithLong: n]));
            },
            [&] (const unsigned long& n) {
                return bridge_cast(adoptNS([[NSNumber alloc] initWithUnsignedLong: n]));
            },
            [&] (const long long& n) {
                return bridge_cast(adoptNS([[NSNumber alloc] initWithLongLong: n]));
            },
            [&] (const unsigned long long& n) {
                return bridge_cast(adoptNS([[NSNumber alloc] initWithUnsignedLongLong: n]));
            },
            [&] (const float& n) {
                return bridge_cast(adoptNS([[NSNumber alloc] initWithFloat: n]));
            },
            [&] (const double& n) {
                return bridge_cast(adoptNS([[NSNumber alloc] initWithDouble: n]));
            }
        );
    }

    CoreIPCNumber::NumberHolder get() const
    {
        return m_numberHolder;
    }

    RetainPtr<id> toID() const { return bridge_cast(createCFNumber().get()); }

private:
    NumberHolder m_numberHolder;
};

}

#endif // USE(CF)
