/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 29, 2023.
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

//
//  HIDEventBase.m
//  iohidobjc
//
//  Created by dekom on 9/11/18.
//

#import <CoreFoundation/CoreFoundation.h>
#import <IOKit/hid/IOHIDEventPrivate.h>
#import <IOKit/hid/IOHIDEventData.h>
#import <IOKit/hid/IOHIDLibPrivate.h>
#import "HIDEventBasePrivate.h"

@implementation HIDEvent

- (CFTypeID)_cfTypeID {
    return IOHIDEventGetTypeID();
}

- (instancetype)initWithSize:(NSUInteger)size
                        type:(IOHIDEventType)type
                   timestamp:(uint64_t)timestamp
                     options:(uint32_t)options
{
    self = [super init];
    
    if (!self || size < sizeof(IOHIDEventData)) {
        return nil;
    }
    
    _event.eventData = (IOHIDEventData *)malloc(size);
    if (_event.eventData == NULL) {
        return nil;
    }
    
    bzero(_event.eventData, size);
    
    _event.timeStamp = timestamp;
    _event.options = options;
    _event.typeMask = IOHIDEventTypeMask(type);
    _event.eventData->type = type;
    _event.eventData->size = (uint32_t)size;
    _event.eventData->options = options;
    
    return self;
}

- (void)dealloc
{
    if (_event.eventData) {
        free(_event.eventData);
    }
    
    if (_event.children) {
        CFIndex count = CFArrayGetCount(_event.children);
        
        for (CFIndex i = 0; i < count; i++) {
            HIDEvent *event;
            
            event = (__bridge HIDEvent *)CFArrayGetValueAtIndex(_event.children, i);
            if (event) {
                event->_event.parent = NULL;
            }
        }
        CFRelease(_event.children);
    }
    
    if (_event.attributeData) {
        free(_event.attributeData);
    }
    
    if (_event.sender) {
        CFRelease(_event.sender);
    }
    
    if (_event.attachments) {
        CFRelease(_event.attachments);
    }
    
    [super dealloc];
}

- (NSString *)description
{
    NSString *desc = (__bridge NSString *)IOHIDEventCopyDescription(
                                                (__bridge IOHIDEventRef)self);
    return [desc autorelease];
}

@end
