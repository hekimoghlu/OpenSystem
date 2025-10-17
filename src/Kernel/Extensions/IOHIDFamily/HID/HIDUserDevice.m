/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 19, 2024.
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
#import <HID/HIDUserDevice.h>
#import <IOKit/hid/IOHIDUserDevice.h>
#import <HID/NSError+IOReturn.h>
#import <os/assumes.h>

NSString * const kHIDUserDevicePropertyCreateInactiveKey = @"HIDUserDeviceCreateInactive";

@implementation HIDUserDevice {
    IOHIDUserDeviceRef              _device;
    HIDUserDeviceGetReportHandler   _getReportHandler;
    HIDUserDeviceSetReportHandler   _setReportHandler;
    io_service_t                    _service;
    HIDBlock                        _cancelHandler;
}

- (instancetype)initWithProperties:(NSDictionary *)properties
{
    IOOptionBits options = 0;
    
    self = [super init];
    
    if (!self) {
        return self;
    }
    
    if (properties[kHIDUserDevicePropertyCreateInactiveKey] && [properties[kHIDUserDevicePropertyCreateInactiveKey] isEqual:@YES]) {
        options |= kIOHIDUserDeviceCreateOptionStartWhenScheduled;
    }
    
    _device = IOHIDUserDeviceCreateWithOptions(kCFAllocatorDefault,
                                               (__bridge CFDictionaryRef)properties,
                                               options
                                               );
    if (!_device) {
        return nil;
    }
    
    _service = IOHIDUserDeviceCopyService(_device);
    
    return self;
}

- (void)dealloc
{
    if (_service) {
        IOObjectRelease(_service);
    }
    if (_device) {
        CFRelease(_device);
    }
}

- (NSString *)description {
    return [NSString stringWithFormat:@"%@", _device];
}

- (id)propertyForKey:(NSString *)key
{
    return (id)CFBridgingRelease(IOHIDUserDeviceCopyProperty(
                                                    _device,
                                                    (__bridge CFStringRef)key));
}

- (BOOL)setProperty:(id)value forKey:(NSString *)key
{
    return IOHIDUserDeviceSetProperty(_device,
                                      (__bridge CFStringRef)key,
                                      (__bridge CFTypeRef)value);
}

- (void)setCancelHandler:(HIDBlock)handler
{
    _cancelHandler = handler;
}

- (void)setDispatchQueue:(dispatch_queue_t)queue
{
    IOHIDUserDeviceSetDispatchQueue(_device, queue);
}

- (void)activate
{
    IOHIDUserDeviceSetCancelHandler(_device, ^{
        // Block captures reference to self while cancellation hasn't completed.
        if (self->_cancelHandler) {
            self->_cancelHandler();
            self->_cancelHandler = nil;
        }
    });

    IOHIDUserDeviceActivate(_device);
}

- (void)cancel
{
    IOHIDUserDeviceCancel(_device);
}

static IOReturn getReportCallback(void *refcon, IOHIDReportType type,
                                  uint32_t reportID, uint8_t *report,
                                  CFIndex *reportLength)
{
    NSInteger length = (NSInteger)*reportLength;
    IOReturn ret;
    HIDUserDevice *me = (__bridge HIDUserDevice *)refcon;
    
    ret = (me->_getReportHandler)((HIDReportType)type,
                                  reportID,
                                  (void *)report,
                                  &length);
    
    *reportLength = (CFIndex)MIN(*reportLength, length);
    
    return ret;
}

- (void)setGetReportHandler:(HIDUserDeviceGetReportHandler)handler
{
    os_assert(!_getReportHandler, "Get report handler already set");
    _getReportHandler = handler;
    
    IOHIDUserDeviceRegisterGetReportWithReturnLengthCallback(
                                                        _device,
                                                        getReportCallback,
                                                        (__bridge void *)self);
}

static IOReturn setReportCallback(void *refcon, IOHIDReportType type,
                                  uint32_t reportID, uint8_t *report,
                                  CFIndex reportLength)
{
    HIDUserDevice *me = (__bridge HIDUserDevice *)refcon;
    
    return (me->_setReportHandler)((HIDReportType)type,
                                   reportID,
                                   (const void *)report,
                                   (NSInteger)reportLength);
}

- (void)setSetReportHandler:(HIDUserDeviceSetReportHandler)handler
{
    os_assert(!_setReportHandler, "Set report handler already set");
    _setReportHandler = handler;
    IOHIDUserDeviceRegisterSetReportCallback(_device,
                                             setReportCallback,
                                             (__bridge void *)self);
}

- (BOOL)handleReport:(NSData *)report
               error:(out NSError **)outError
{
    IOReturn ret = IOHIDUserDeviceHandleReport(_device,
                                               (uint8_t *)[report bytes],
                                               [report length]);
    
    if (ret != kIOReturnSuccess && outError) {
        *outError = [NSError errorWithIOReturn:ret];
    }
    
    return (ret == kIOReturnSuccess);
}

- (BOOL)handleReport:(NSData *)report
       withTimestamp:(uint64_t)timestamp
               error:(out NSError **)outError
{
    IOReturn ret = IOHIDUserDeviceHandleReportWithTimeStamp(_device,
                                                            timestamp,
                                                            (uint8_t *)[report bytes],
                                                            [report length]);

    if (ret != kIOReturnSuccess && outError) {
        *outError = [NSError errorWithIOReturn:ret];
    }

    return (ret == kIOReturnSuccess);
}

- (io_service_t)service
{
    return _service;
}

@end
