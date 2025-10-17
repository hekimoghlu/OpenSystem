/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 3, 2023.
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
/*
 * Copyright (c) 1997 Apple Computer, Inc.
 *
 *
 * HISTORY
 *
 */


#ifndef _IOKIT_IOHIDUSERCLIENT_H
#define _IOKIT_IOHIDUSERCLIENT_H

#include <libkern/c++/OSContainers.h>
#include <IOKit/IOUserClient.h>
#include "IOHIDSystem.h"
#include "IOHIDEventServiceQueue.h"

#define MAX_SCREENS 64  // same as EV_MAX_SCREENS in HIDSystem

class IOHIDUserClient : public IOUserClient2022
{
    OSDeclareDefaultStructors(IOHIDUserClient)

private:

    IOHIDSystem     *owner;
    int             _screenTokens[MAX_SCREENS];

public:
    // IOUserClient methods
    virtual IOReturn clientClose( void ) APPLE_KEXT_OVERRIDE;

    virtual IOService * getService( void ) APPLE_KEXT_OVERRIDE;

    virtual IOReturn registerNotificationPort(
		mach_port_t port, UInt32 type, UInt32 refCon ) APPLE_KEXT_OVERRIDE;

    virtual IOReturn connectClient( IOUserClient * client ) APPLE_KEXT_OVERRIDE;

    virtual IOReturn clientMemoryForType( UInt32 type,
        UInt32 * flags, IOMemoryDescriptor ** memory ) APPLE_KEXT_OVERRIDE;

    virtual IOReturn externalMethod(uint32_t selector, IOExternalMethodArgumentsOpaque * args) APPLE_KEXT_OVERRIDE;

    virtual IOExternalMethod * getTargetAndMethodForIndex(
                        IOService ** targetP, UInt32 index ) APPLE_KEXT_OVERRIDE;

    // others
    virtual bool initWithTask(task_t owningTask, void * /* security_id */, UInt32 /* type */) APPLE_KEXT_OVERRIDE;
    virtual bool start( IOService * provider ) APPLE_KEXT_OVERRIDE;
    virtual void stop( IOService * provider ) APPLE_KEXT_OVERRIDE;
    virtual IOReturn close( void );
    
    virtual IOReturn setProperties( OSObject * properties ) APPLE_KEXT_OVERRIDE;
    IOReturn extGetUserHidActivityState(void*,void*,void*,void*,void*,void*);
};


class IOHIDParamUserClient : public IOUserClient2022
{
    OSDeclareDefaultStructors(IOHIDParamUserClient)

private:

    IOHIDSystem     *owner;
    
public:

    // IOUserClient methods    
    virtual IOService * getService( void ) APPLE_KEXT_OVERRIDE;

    virtual IOReturn externalMethod(uint32_t selector, IOExternalMethodArgumentsOpaque * args) APPLE_KEXT_OVERRIDE;

    virtual IOExternalMethod * getTargetAndMethodForIndex(
                        IOService ** targetP, UInt32 index ) APPLE_KEXT_OVERRIDE;

    // others

    virtual bool start( IOService * provider ) APPLE_KEXT_OVERRIDE;
    virtual IOReturn setProperties( OSObject * properties ) APPLE_KEXT_OVERRIDE;

    IOReturn extGetUserHidActivityState(void*,void*,void*,void*,void*,void*);
private:
    virtual IOReturn clientClose(void) APPLE_KEXT_OVERRIDE;
    virtual IOReturn extPostEvent(void*,void*,void*,void*,void*,void*);
};

class IOHIDEventSystemUserClient : public IOUserClient2022
{
    OSDeclareDefaultStructors(IOHIDEventSystemUserClient)

private:
    IOHIDSystem *               owner;
    IOHIDEventServiceQueue *    kernelQueue;
    IOCommandGate *             commandGate;
    mach_port_t                 _port;
    
    IOReturn registerNotificationPortGated(mach_port_t port, UInt32 type, UInt32 refCon);

public:
    virtual bool initWithTask(task_t owningTask, void * security_id, UInt32 type ) APPLE_KEXT_OVERRIDE;
    void free(void) APPLE_KEXT_OVERRIDE;

    // IOUserClient methods    
    virtual IOReturn clientClose( void ) APPLE_KEXT_OVERRIDE;

    virtual IOReturn externalMethod(uint32_t selector, IOExternalMethodArgumentsOpaque * args) APPLE_KEXT_OVERRIDE;
    virtual IOExternalMethod * getTargetAndMethodForIndex(IOService ** targetP, UInt32 index ) APPLE_KEXT_OVERRIDE;
    virtual IOReturn createEventQueue(void*,void*,void*,void*,void*,void*);
    virtual IOReturn createEventQueueGated(void*p1,void*p2,void*p3, void*);
    virtual IOReturn destroyEventQueue(void*,void*,void*,void*,void*,void*);
    virtual IOReturn destroyEventQueueGated(void*,void*,void*,void*);
    virtual IOReturn tickle(void*,void*,void*,void*,void*,void*);

    virtual IOReturn registerNotificationPort(mach_port_t port, UInt32 type, UInt32 refCon ) APPLE_KEXT_OVERRIDE;
    virtual IOReturn clientMemoryForType( UInt32 type, UInt32 * flags, IOMemoryDescriptor ** memory ) APPLE_KEXT_OVERRIDE;
    IOReturn clientMemoryForTypeGated( UInt32 type, UInt32 * flags, IOMemoryDescriptor ** memory );

    virtual IOService * getService( void ) APPLE_KEXT_OVERRIDE;

    virtual bool start( IOService * provider ) APPLE_KEXT_OVERRIDE;
    virtual void stop ( IOService * provider ) APPLE_KEXT_OVERRIDE;
};



#endif /* ! _IOKIT_IOHIDUSERCLIENT_H */
