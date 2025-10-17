/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 28, 2023.
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
#ifndef IOHIDDeviceClass_h
#define IOHIDDeviceClass_h

#import "IOHIDIUnknown2.h"
#import <IOKit/hid/IOHIDElement.h>
#import <IOKit/hid/IOHIDValue.h>
#import <IOKit/hid/IOHIDDevicePlugIn.h>
#import <IOKit/hid/IOHIDLibUserClient.h>
#import <os/lock_private.h>

@class IOHIDQueueClass;

enum {
    kHIDSetElementValuePendEvent    = 0x00010000,
    kHIDGetElementValuePendEvent    = kHIDSetElementValuePendEvent,
    kHIDGetElementValueForcePoll    = 0x00020000,
    kHIDGetElementValuePreventPoll  = 0x00040000,
};

enum {
    kHIDCopyMatchingElementsDictionary = 0x1
};

@interface IOHIDDeviceClass : IOHIDPlugin {
    IOHIDDeviceTimeStampedDeviceInterface   *_device;
    io_service_t                            _service;
    io_connect_t                            _connect;

    os_unfair_recursive_lock                _deviceLock;
    
    mach_port_t                             _port;
    CFMachPortRef                           _machPort;
    CFRunLoopSourceRef                      _runLoopSource;
    
    BOOL                                    _opened;
    BOOL                                    _tccRequested;
    BOOL                                    _tccGranted;
    
    IOHIDQueueClass                         *_queue;
    NSMutableArray                          *_elements;
    NSMutableArray                          *_sortedElements;
    NSMutableArray                          *_reportElements;
    NSMutableDictionary                     *_properties;
    
    os_unfair_recursive_lock                _callbackLock;
    IOHIDReportCallback                     _inputReportCallback;
    IOHIDReportWithTimeStampCallback        _inputReportTimestampCallback;
    void                                    *_inputReportContext;
    uint8_t                                 *_inputReportBuffer;
    CFIndex                                 _inputReportBufferLength;
    NSDictionary                            *_protectedEvent;
}

- (mach_port_t)getPort;
- (void)initQueue;

- (IOReturn)open:(IOOptionBits)options;
- (IOReturn)close:(IOOptionBits)options;

- (IOReturn)copyMatchingElements:(NSDictionary * _Nullable)matching
                        elements:(CFArrayRef _Nonnull * _Nonnull)pElements
                         options:(IOOptionBits)options;

- (IOReturn)setInputReportCallback:(uint8_t * _Nonnull)report
                      reportLength:(CFIndex)reportLength
                          callback:(IOHIDReportCallback _Nonnull)callback
                           context:(void * _Nullable)context
                           options:(IOOptionBits)options;

- (IOReturn)setReport:(IOHIDReportType)reportType
             reportID:(uint32_t)reportID
               report:(const uint8_t * _Nonnull)report
         reportLength:(CFIndex)reportLength
              timeout:(uint32_t)timeout
             callback:(IOHIDReportCallback _Nullable)callback
              context:(void * _Nullable)context
              options:(IOOptionBits)options;

- (IOReturn)getReport:(IOHIDReportType)reportType
             reportID:(uint32_t)reportID
               report:(uint8_t * _Nonnull)report
         reportLength:(CFIndex * _Nonnull)pReportLength
              timeout:(uint32_t)timeout
             callback:(IOHIDReportCallback _Nullable)callback
              context:(void * _Nullable)context
              options:(IOOptionBits)options;

- (IOReturn)getValue:(IOHIDElementRef _Nonnull)element
               value:(IOHIDValueRef _Nonnull * _Nonnull)pValue
             timeout:(uint32_t)timeout
            callback:(IOHIDValueCallback _Nullable)callback
             context:(void * _Nullable)context
             options:(IOOptionBits)options;

- (IOReturn)setValue:(IOHIDElementRef _Nonnull)element
               value:(IOHIDValueRef _Nonnull)value
             timeout:(uint32_t)timeout
            callback:(IOHIDValueCallback _Nullable)callback
             context:(void * _Nullable)context
             options:(IOOptionBits)options;

- (IOHIDElementRef _Nullable)getElement:(uint32_t)cookie;

- (void)releaseReport:(uint64_t)reportAddress;

@property (readonly)            mach_port_t         port;
@property (readonly, nullable)  CFRunLoopSourceRef  runLoopSource;
@property (readonly)            io_connect_t        connect;
@property (readonly)            io_service_t        service;

@end

#endif /* IOHIDDeviceClass_h */
