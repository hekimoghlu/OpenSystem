/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 25, 2022.
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
// powerwatch - hook into system notifications of power events
//
#include "powerwatch.h"
#include <IOKit/IOMessage.h>

#if TARGET_OS_OSX

namespace Security {
namespace MachPlusPlus {


//
// The obligatory empty virtual destructor
//
PowerWatcher::~PowerWatcher()
{ }


//
// The default NULL implementations of the callback virtuals.
// We define these (rather than leaving them abstract) since
// many users want only one of these events.
//
void PowerWatcher::systemWillSleep()
{ }

void PowerWatcher::systemIsWaking()
{ }

void PowerWatcher::systemWillPowerDown()
{ }

void PowerWatcher::systemWillPowerOn()
{ }

//
// IOPowerWatchers
//

IOPowerWatcher::IOPowerWatcher() :
    mKernelPort(0)
{
	if (!(mKernelPort = ::IORegisterForSystemPower(this, &mPortRef, ioCallback, &mHandle)))
		UnixError::throwMe(EINVAL);	// no clue
}

IOPowerWatcher::~IOPowerWatcher()
{
	if (mKernelPort)
		::IODeregisterForSystemPower(&mHandle);
}

//
// The callback dispatcher
//
void IOPowerWatcher::ioCallback(void *refCon, io_service_t service,
    natural_t messageType, void *argument)
{
    IOPowerWatcher *me = (IOPowerWatcher *)refCon;
    enum { allow, refuse, ignore } reaction;
    switch (messageType) {
    case kIOMessageSystemWillSleep:
        secnotice("powerwatch", "system will sleep");
        me->systemWillSleep();
        reaction = allow;
        break;
    case kIOMessageSystemHasPoweredOn:
        secnotice("powerwatch", "system has powered on");
        me->systemIsWaking();
        reaction = ignore;
        break;
    case kIOMessageSystemWillPowerOff:
        secnotice("powerwatch", "system will power off");
        me->systemWillPowerDown();
        reaction = allow;
        break;
    case kIOMessageSystemWillNotPowerOff:
        secnotice("powerwatch", "system will not power off");
        reaction = ignore;
        break;
    case kIOMessageCanSystemSleep:
        secnotice("powerwatch", "can system sleep");
        reaction = allow;
        break;
    case kIOMessageSystemWillNotSleep:
        secnotice("powerwatch", "system will not sleep");
        reaction = ignore;
        break;
    case kIOMessageCanSystemPowerOff:
        secnotice("powerwatch", "can system power off");
        reaction = allow;
        break;
	case kIOMessageSystemWillPowerOn:
        secnotice("powerwatch", "system will power on");
		me->systemWillPowerOn();
        reaction = ignore;
        break;
    default:
        secnotice("powerwatch",
            "type 0x%x message received (ignored)", messageType);
        reaction = ignore;
        break;
    }
    
    // handle acknowledgments
    switch (reaction) {
    case allow:
		secnotice("powerwatch", "calling IOAllowPowerChange");
        IOAllowPowerChange(me->mKernelPort, long(argument));
        break;
    case refuse:
		secnotice("powerwatch", "calling IOCancelPowerChange");
        IOCancelPowerChange(me->mKernelPort, long(argument));
        break;
    case ignore:
		secnotice("powerwatch", "sending no response");
        break;
    }
}


//
// The MachServer hookup
//
PortPowerWatcher::PortPowerWatcher()
{
    port(IONotificationPortGetMachPort(mPortRef));
}

boolean_t PortPowerWatcher::handle(mach_msg_header_t *in)
{
    IODispatchCalloutFromMessage(NULL, in, mPortRef);
    return TRUE;
}


} // end namespace MachPlusPlus

} // end namespace Security

#endif //TARGET_OS_OSX
