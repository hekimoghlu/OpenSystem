/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 25, 2023.
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
//  IOHIDNXEventTranslatorServiceFilter.h
//  IOHIDFamily
//
//  Created by Yevgen Goryachok 11/04/15.
//
//

#ifndef _IOHIDFamily_IOHIDNXEventTranslatorServiceFilter_
#define _IOHIDFamily_IOHIDNXEventTranslatorServiceFilter_
#include <sys/cdefs.h>
#include <CoreFoundation/CoreFoundation.h>
#if COREFOUNDATION_CFPLUGINCOM_SEPARATE
#include <CoreFoundation/CFPlugInCOM.h>
#endif

#include <IOKit/hid/IOHIDService.h>
#include <IOKit/hid/IOHIDServiceFilterPlugIn.h>
#include <IOKit/hid/IOHIDUsageTables.h>
#include "IOHIDEventTranslation.h"

#define kHIDEventTranslationSupport         "HIDEventTranslationSupport"
#define kHIDEventTranslationModifierFlags   "HIDEventTranslationModifierFlags"

class IOHIDNXEventTranslatorServiceFilter
{
public:
    IOHIDNXEventTranslatorServiceFilter(CFUUIDRef factoryID);
    ~IOHIDNXEventTranslatorServiceFilter();
    HRESULT QueryInterface( REFIID iid, LPVOID *ppv );
    ULONG AddRef();
    ULONG Release();
    
    SInt32 match(IOHIDServiceRef service, IOOptionBits options);
    IOHIDEventRef filter(IOHIDEventRef event);
    void open(IOHIDServiceRef session, IOOptionBits options);
    void close(IOHIDServiceRef session, IOOptionBits options);
    void registerService(IOHIDServiceRef service);
    void handlePendingStats();
    void scheduleWithDispatchQueue(dispatch_queue_t queue);
    void unscheduleFromDispatchQueue(dispatch_queue_t queue);
    CFTypeRef copyPropertyForClient(CFStringRef key, CFTypeRef client);
    void setPropertyForClient(CFStringRef key, CFTypeRef property, CFTypeRef client);
    void setEventCallback(IOHIDServiceEventCallback callback, void * target, void * refcon);
    
private:
    static IOHIDServiceFilterPlugInInterface  sIOHIDEventSystemStatisticsFtbl;
    IOHIDServiceFilterPlugInInterface         *_serviceInterface;
    CFUUIDRef                                 _factoryID;
    UInt32                                    _refCount;
    SInt32                                    _matchScore;
 
    static IOHIDServiceFilterPlugInInterface sIOHIDNXEventTranslatorServiceFilterFtbl;
    static HRESULT QueryInterface( void *self, REFIID iid, LPVOID *ppv );
    static ULONG AddRef( void *self );
    static ULONG Release( void *self );
    
    static SInt32 match(void * self, IOHIDServiceRef service, IOOptionBits options);
    static IOHIDEventRef filter(void * self, IOHIDEventRef event);
    
    static void open(void * self, IOHIDServiceRef inService, IOOptionBits options);
    static void close(void * self, IOHIDServiceRef inSession, IOOptionBits options);
    
    static void scheduleWithDispatchQueue(void * self, dispatch_queue_t queue);
    static void unscheduleFromDispatchQueue(void * self, dispatch_queue_t queue);

    static CFTypeRef copyPropertyForClient(void * self, CFStringRef key, CFTypeRef client);
    static void setPropertyForClient(void * self,CFStringRef key, CFTypeRef property, CFTypeRef client);

    IOHIDServiceEventCallback _eventCallback;
    void * _eventTarget;
    void * _eventContext;
    static void setEventCallback(void * self, IOHIDServiceEventCallback callback, void * target, void * refcon);

    dispatch_queue_t                _queue;
    IOHIDServiceRef                 _service;
    IOHIDKeyboardEventTranslatorRef _translator;
    bool                            _isTranslationEnabled;
  
    void serialize (CFMutableDictionaryRef dict) const;

private:
  
    IOHIDNXEventTranslatorServiceFilter();
    IOHIDNXEventTranslatorServiceFilter(const IOHIDNXEventTranslatorServiceFilter &);
    IOHIDNXEventTranslatorServiceFilter &operator=(const IOHIDNXEventTranslatorServiceFilter &);
};


#endif /* defined(_IOHIDFamily_IOHIDNXEventTranslatorServiceFilter_) */
