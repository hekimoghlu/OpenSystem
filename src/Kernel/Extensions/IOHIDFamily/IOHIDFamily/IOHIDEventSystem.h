/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 4, 2022.
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
#ifndef _IOKIT_HID_IOHIDEVENTSYSTEM_H
#define _IOKIT_HID_IOHIDEVENTSYSTEM_H

#include <IOKit/IOMessage.h>
#include <IOKit/IOWorkLoop.h>
#include <IOKit/IOCommandGate.h>
#include <IOKit/IOTimerEventSource.h>
#include "IOHIDEventService.h"
#include "IOHIDEventTypes.h"

class IOHIDEventSystem: public IOService
{
    OSDeclareAbstractStructors( IOHIDEventSystem )

    IOWorkLoop *		_workLoop;
    IOCommandGate *     _commandGate;
    
    IONotifier *		_publishNotify;
    IONotifier *		_terminateNotify;
    OSArray *           _eventServiceInfoArray;
    
    bool                _eventsOpen;
    
    struct ExpansionData { 
    };
    /*! @var reserved
        Reserved for future use.  (Internal use only)  */
    ExpansionData *         _reserved;

    bool notificationHandler(
                                void *                          refCon, 
                                IOService *                     service );
                                
    void handleHIDEvent(
                                void *                          refCon,
                                AbsoluteTime                    timeStamp,
                                UInt32                          eventCount,
                                IOHIDEvent *                    events,
                                IOOptionBits                    options);    
    

    // Gated Methods
    void handleServicePublicationGated(IOService * service);

    void handleServiceTerminationGated(IOService * service);

    void handleHIDEventGated(void * args);
    
    void registerEventSource(IOHIDEventService * service);

public:
    virtual bool      init(OSDictionary * properties = 0);
    virtual bool      start(IOService * provider);
    virtual void      free();
    virtual IOReturn  message(UInt32 type, IOService * provider, void * argument);
    virtual IOReturn  setProperties( OSObject * properties );

};

#endif /* _IOKIT_HID_IOHIDEVENTSYSTEM_H */
