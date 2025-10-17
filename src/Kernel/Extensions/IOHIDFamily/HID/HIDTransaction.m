/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 5, 2024.
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
#import <IOKit/hid/IOHIDLib.h>
#import <HID/HIDTransaction.h>
#import <HID/HIDDevice.h>
#import <HID/HIDElement_Internal.h>
#import <HID/NSError+IOReturn.h>
#import <IOKit/hid/IOHIDLibPrivate.h>

@implementation HIDTransaction {
    IOHIDTransactionRef _transaction;
}

- (instancetype)initWithDevice:(HIDDevice *)device
{
    self = [super init];
    
    if (!self) {
        return self;
    }
    
    _transaction = IOHIDTransactionCreate(kCFAllocatorDefault,
                                          (__bridge IOHIDDeviceRef)device,
                                          kIOHIDTransactionDirectionTypeInput,
                                          kIOHIDTransactionOptionsWeakDevice);
    if (!_transaction) {
        return nil;
    }
    
    return self;
}

- (void)dealloc
{
    if (_transaction) {
        CFRelease(_transaction);
    }
}

- (NSString *)description {
    return [NSString stringWithFormat:@"%@", _transaction];
}

- (HIDTransactionDirectionType)direction
{
    return (HIDTransactionDirectionType)IOHIDTransactionGetDirection(_transaction);
}

- (void)setDirection:(HIDTransactionDirectionType)direction
{
    IOHIDTransactionSetDirection(_transaction,
                                 (IOHIDTransactionDirectionType)direction);
}

typedef void (^CommitCallbackInternal)(IOReturn status);

static void asyncCommitCallback(void   * context,
                                IOReturn result,
                                void   * sender __unused)
{
    HIDTransactionCommitCallback callback = (__bridge HIDTransactionCommitCallback)context;

    callback(result);

    Block_release(context);
}

- (BOOL)commitElements:(NSArray<HIDElement *> *)elements
                 error:(out NSError **)outError
{
    return [self commitElements:elements error:outError timeout:0 callback:nil];
}

- (BOOL)commitElements:(NSArray<HIDElement *> *)elements
                 error:(out NSError **)outError
               timeout:(NSInteger)timeout
              callback:(HIDTransactionCommitCallback _Nullable)callback
{
    IOReturn ret = kIOReturnError;
    
    for (HIDElement * element in elements) {
        IOHIDTransactionAddElement(_transaction, (__bridge IOHIDElementRef)element);
    }

    if (callback) {
        ret = IOHIDTransactionCommitWithCallback(_transaction, timeout, asyncCommitCallback, Block_copy((__bridge void *)callback));
    } else {
        ret = IOHIDTransactionCommit(_transaction);
    }
    
    if (ret != kIOReturnSuccess) {
        IOHIDTransactionClear(_transaction);
        
        if (outError) {
            *outError = [NSError errorWithIOReturn:ret];
        }

        if (callback) {
            Block_release((__bridge void *)callback);
        }
        
        return NO;
    }

    IOHIDTransactionClear(_transaction);
    return (ret == kIOReturnSuccess);
}

@end
