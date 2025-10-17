/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 19, 2022.
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
#ifndef IOHIDUPSClass_h
#define IOHIDUPSClass_h

#import "IOHIDIUnknown2.h"
#import <IOKit/ps/IOUPSPlugIn.h>
#import <IOKit/hid/IOHIDDevicePlugIn.h>

@interface IOHIDUPSClass : IOHIDPlugin {
    IOUPSPlugInInterface_v140               *_ups;
    IOHIDDeviceTimeStampedDeviceInterface   **_device;
    IOHIDDeviceQueueInterface               **_queue;
    IOHIDDeviceTransactionInterface         **_transaction;
    
    NSMutableDictionary                     *_properties;
    NSMutableSet                            *_capabilities;
    NSMutableDictionary                     *_upsEvent;
    NSMutableDictionary                     *_upsUpdatedEvent;
    NSMutableDictionary                     *_debugInformation;
    
    struct {
        NSMutableArray                      *input;
        NSMutableArray                      *output;
        NSMutableArray                      *feature;
    } _elements;
    
    NSMutableArray                          *_commandElements;
    NSMutableArray                          *_eventElements;
    
    IOUPSEventCallbackFunction              _eventCallback;
    void *                                  _eventTarget;
    void *                                  _eventRefcon;
    
    NSTimer                                 *_timer;
    CFRunLoopSourceRef                      _runLoopSource;
}

@end

#endif /* IOHIDUPSClass_h */
