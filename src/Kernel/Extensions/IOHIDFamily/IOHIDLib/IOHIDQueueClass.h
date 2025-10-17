/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 17, 2024.
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
#ifndef IOHIDQueueClass_h
#define IOHIDQueueClass_h

#import "IOHIDIUnknown2.h"
#import <IOKit/hid/IOHIDDevicePlugIn.h>
#import <IOKit/IODataQueueShared.h>
#import <os/lock_private.h>
#import "IOHIDDeviceClass.h"

@interface IOHIDQueueClass : IOHIDIUnknown2 {
    IOHIDDeviceQueueInterface   *_queue;
    __weak IOHIDDeviceClass     *_device;
    
    os_unfair_lock              _queueLock;

    mach_port_t                 _port;
    CFMachPortRef               _machPort;
    CFRunLoopSourceRef          _runLoopSource;
    
    IOHIDQueueHeader            *_queueHeader;
    IODataQueueMemory           *_queueMemory;
    vm_size_t                   _queueMemorySize;
    bool                        _queueSizeChanged;
    uint32_t                    _lastTail;
    
    uint32_t                    _depth;
    uint64_t                    _queueToken;
    
    IOHIDCallback               _valueAvailableCallback;
    void                        *_valueAvailableContext;

    CFTypeRef                   _usageAnalytics;
}

- (nullable instancetype)initWithDevice:(IOHIDDeviceClass * _Nonnull)device;
- (nullable instancetype)initWithDevice:(IOHIDDeviceClass * _Nonnull)device
                                   port:(mach_port_t)port
                                 source:(CFRunLoopSourceRef _Nullable)source;

- (IOReturn)addElement:(IOHIDElementRef _Nonnull)element;
- (IOReturn)setValueAvailableCallback:(IOHIDCallback _Nonnull)callback
                              context:(void * _Nullable)context;
- (IOReturn)start;
- (IOReturn)stop;
- (IOReturn)copyNextValue:(IOHIDValueRef _Nullable * _Nullable)pValue;

- (void)queueCallback:(CFMachPortRef _Nonnull)port
                  msg:(mach_msg_header_t * _Nonnull)msg
                 size:(CFIndex)size
                 info:(void * _Nullable)info;

- (void)signalQueueEmpty;

@end;

// We will have to support this until kIOHIDDeviceInterfaceID is deprecated
// (see 35698866)
@interface IOHIDObsoleteQueueClass : IOHIDQueueClass {
    IOHIDQueueInterface     *_interface;
    
    IOHIDCallbackFunction   _eventCallback;
    void                    *_eventCallbackTarget;
    void                    *_eventCallbackRefcon;
}

@end;

#endif /* IOHIDQueueClass_h */
