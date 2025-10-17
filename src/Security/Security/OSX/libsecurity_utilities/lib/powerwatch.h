/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 8, 2024.
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
#ifndef _H_POWERWATCH
#define _H_POWERWATCH

#include <TargetConditionals.h>

#if TARGET_OS_OSX

#include <security_utilities/machserver.h>
#include <IOKit/pwr_mgt/IOPMLib.h>


namespace Security {
namespace MachPlusPlus {


//
// PowerWatcher embodies the ability to respond to power events.
// By itself, it is inert - nobody will call its virtual methods.
// Use one of it subclasses, which take care of "hooking" into an
// event delivery mechanism.
//
class PowerWatcher {
public:
    virtual ~PowerWatcher();
    
public:
    virtual void systemWillSleep();
    virtual void systemIsWaking();
    virtual void systemWillPowerDown();
	virtual void systemWillPowerOn();
};


//
// A PowerWatcher that is dispatches events from an IOKit message
//
class IOPowerWatcher : public PowerWatcher {
public:
	IOPowerWatcher();
	~IOPowerWatcher();
    
protected:
    io_connect_t mKernelPort;
    IONotificationPortRef mPortRef;
    io_object_t mHandle;

    static void ioCallback(void *refCon, io_service_t service,
        natural_t messageType, void *argument);

};


//
// Hook into a "raw" MachServer object for event delivery
//
class PortPowerWatcher : public IOPowerWatcher, public MachServer::NoReplyHandler {
public:
    PortPowerWatcher();
    
    boolean_t handle(mach_msg_header_t *in);    
};


//
// Someone should add a RunLoopPowerWatcher class here, I suppose.
// Well, if you need one: Tag, You're It!
//


} // end namespace MachPlusPlus

} // end namespace Security

#endif //TARGET_OS_OSX

#endif //_H_POWERWATCH
