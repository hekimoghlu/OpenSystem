/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 1, 2025.
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
#ifndef _IOKIT_IOHIDNXEVENTROUTER_H
#define _IOKIT_IOHIDNXEVENTROUTER_H

#include <dispatch/dispatch.h>
#include <IOKit/hid/IOHIDServicePlugIn.h>
#include <IOKit/IODataQueueClient.h>
#include "IOHIDIUnknown.h"
#include <IOKit/hid/IOHIDEventQueue.h>

class IOHIDNXEventRouter : public IOHIDIUnknown
{
private:
    // Disable copy constructors
    IOHIDNXEventRouter(IOHIDNXEventRouter &src);
    void operator =(IOHIDNXEventRouter &src);

protected:
    IOHIDNXEventRouter();
    virtual ~IOHIDNXEventRouter();

    static IOCFPlugInInterface          sIOCFPlugInInterfaceV1;
    static IOHIDServiceInterface2       sIOHIDServiceInterface2;

    struct InterfaceMap                 _hidService;
    io_service_t                        _service;
    bool                                _isOpen;
    
    mach_port_t                         _asyncPort;

    dispatch_source_t                   _asyncEventSource;
    
    CFMutableDictionaryRef              _serviceProperties;
    CFMutableDictionaryRef              _servicePreferences;
        
    IOHIDServiceEventCallback           _eventCallback;
    void *                              _eventTarget;
    void *                              _eventRefcon;

  
    dispatch_queue_t                    _dispatchQueue;
    IOHIDEventQueueRef                  queue_;
  
    static inline IOHIDNXEventRouter *getThis(void *self) { return (IOHIDNXEventRouter *)((InterfaceMap *) self)->obj; };

    // IOCFPlugInInterface methods
    static IOReturn _probe(void *self, CFDictionaryRef propertyTable, io_service_t service, SInt32 *order);
    static IOReturn _start(void *self, CFDictionaryRef propertyTable, io_service_t service);
    static IOReturn _stop(void *self);

    // IOHIDServiceInterface2 methods
    static boolean_t        _open(void *self, IOOptionBits options);
    static void             _close(void *self, IOOptionBits options);
    static CFTypeRef        _copyProperty(void *self, CFStringRef key);
    static boolean_t        _setProperty(void *self, CFStringRef key, CFTypeRef property);
    static IOHIDEventRef    _copyEvent(void *self, IOHIDEventType type, IOHIDEventRef matching, IOOptionBits options);
    static IOReturn         _setOutputEvent(void *self, IOHIDEventRef event);
    static void             _setEventCallback(void *self, IOHIDServiceEventCallback callback, void * target, void * refcon);
    static void             _scheduleWithDispatchQueue(void *self, dispatch_queue_t queue);
    static void             _unscheduleFromDispatchQueue(void *self, dispatch_queue_t queue);
    
    // Support methods
    static void             _queueEventSourceCallback(void * info);
    void                    dispatchHIDEvent(IOHIDEventRef event, IOOptionBits options=0);
    static void             _queueCallback       (void * info);
  
public:
    // IOCFPlugin stuff
    static IOCFPlugInInterface **alloc();

    virtual HRESULT         queryInterface(REFIID iid, void **ppv);
    virtual IOReturn        probe(CFDictionaryRef propertyTable, io_service_t service, SInt32 * order);
    virtual IOReturn        start(CFDictionaryRef propertyTable, io_service_t service);
    virtual IOReturn        stop();
    
    virtual boolean_t       open(IOOptionBits options);
    virtual void            close(IOOptionBits options);
    virtual CFTypeRef       copyProperty(CFStringRef key);
    virtual boolean_t       setProperty(CFStringRef key, CFTypeRef property);
    virtual IOHIDEventRef   copyEvent(IOHIDEventType type, IOHIDEventRef matching, IOOptionBits options);
    virtual IOReturn        setOutputEvent(IOHIDEventRef event);
    virtual void            setEventCallback(IOHIDServiceEventCallback callback, void * target, void * refcon);
    virtual void            scheduleWithDispatchQueue(dispatch_queue_t queue);
    virtual void            unscheduleFromDispatchQueue(dispatch_queue_t queue);
};

#endif /* !_IOKIT_IOHIDNXEVENTROUTER_H */
