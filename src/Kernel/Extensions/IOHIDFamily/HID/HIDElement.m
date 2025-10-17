/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 30, 2024.
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
#import <Foundation/Foundation.h>
#import <IOKit/hid/IOHIDLibPrivate.h>
#import <HID/HIDElement_Internal.h>

@implementation HIDElement (HIDFramework)

- (NSString *)description {
    return [NSString stringWithFormat:@"timestamp: %llu type: %ld \
usagePage: %ld usage: %ld reportID: %ld value: %ld",
            self.timestamp, (long)self.type, (long)self.usagePage,
            (long)self.usage, (long)self.reportID, (long)self.integerValue];
}

- (IOHIDValueRef)valueRef
{
    return _IOHIDElementGetValue((__bridge IOHIDElementRef)self);
}

-(void)setValueRef:(IOHIDValueRef)valueRef
{
    _IOHIDElementSetValue((__bridge IOHIDElementRef)self, valueRef);
}

- (uint32_t)cookie
{
    return (uint32_t)IOHIDElementGetCookie((__bridge IOHIDElementRef)self);
}

- (double)scaleValue:(HIDValueScaleType)type
{
    return self.valueRef ? IOHIDValueGetScaledValue(
                                                self.valueRef,
                                                (IOHIDValueScaleType)type) : 0;
}

- (NSInteger)integerValue
{
    return self.valueRef ? IOHIDValueGetIntegerValue(self.valueRef) : 0;
}

- (void)setIntegerValue:(NSInteger)integerValue
{
    IOHIDValueRef value = IOHIDValueCreateWithIntegerValue(
                                                kCFAllocatorDefault,
                                                (__bridge IOHIDElementRef)self,
                                                0,
                                                integerValue);
    if (value) {
        self.valueRef = value;
        CFRelease(value);
    }
}

- (nullable NSData *)dataValue
{
    return self.valueRef ? [NSData dataWithBytes:(void *)IOHIDValueGetBytePtr(self.valueRef)
                                          length:IOHIDValueGetLength(self.valueRef)] : nil;
}

- (void)setDataValue:(NSData *)dataValue
{
    IOHIDValueRef value = IOHIDValueCreateWithBytes(
                                                kCFAllocatorDefault,
                                                (__bridge IOHIDElementRef)self,
                                                0,
                                                [dataValue bytes],
                                                [dataValue length]);
    if (value) {
        self.valueRef = value;
        CFRelease(value);
    }
}

- (HIDElement *)parent
{
    return (__bridge HIDElement *)IOHIDElementGetParent(
                                                (__bridge IOHIDElementRef)self);
}

- (NSArray<HIDElement *> *)children
{
    return (__bridge NSArray *)IOHIDElementGetChildren(
                                                (__bridge IOHIDElementRef)self);
}

- (HIDElementType)type
{
    return (HIDElementType)IOHIDElementGetType((__bridge IOHIDElementRef)self);
}

- (NSInteger)usagePage
{
    return IOHIDElementGetUsagePage((__bridge IOHIDElementRef)self);
}

- (NSInteger)usage
{
    return IOHIDElementGetUsage((__bridge IOHIDElementRef)self);
}

- (NSInteger)reportID
{
    return IOHIDElementGetReportID((__bridge IOHIDElementRef)self);
}

- (NSInteger)reportSize
{
    return IOHIDElementGetReportSize((__bridge IOHIDElementRef)self);
}

- (NSInteger)unit
{
    return IOHIDElementGetUnit((__bridge IOHIDElementRef)self);
}

- (NSInteger)unitExponent
{
    return IOHIDElementGetUnitExponent((__bridge IOHIDElementRef)self);
}

- (NSInteger)logicalMin
{
    return IOHIDElementGetLogicalMin((__bridge IOHIDElementRef)self);
}

- (NSInteger)logicalMax
{
    return IOHIDElementGetLogicalMax((__bridge IOHIDElementRef)self);
}

- (NSInteger)physicalMin
{
    return IOHIDElementGetPhysicalMin((__bridge IOHIDElementRef)self);
}

- (NSInteger)physicalMax
{
    return IOHIDElementGetPhysicalMax((__bridge IOHIDElementRef)self);
}

- (uint64_t)timestamp
{
    return self.valueRef ? IOHIDValueGetTimeStamp(self.valueRef) : 0;
}

@end
