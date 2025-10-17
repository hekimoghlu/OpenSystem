/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 17, 2022.
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
 * Copyright (c) 2000 Apple Computer, Inc.  All rights reserved.
 *
 * HISTORY
 * 09 Nov 2000 wgulland created.
 *
 */

// public
#import <IOKit/firewire/IOFireWireDevice.h>
#import <IOKit/firewire/IOFWDCLPool.h>

// protected
#include <IOKit/firewire/IOFireWireLink.h>
#import <IOKit/firewire/IOFWWorkLoop.h>

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

OSDefineMetaClass( IOFireWireLink, IOService )
OSDefineAbstractStructors(IOFireWireLink, IOService)

OSMetaClassDefineReservedUnused(IOFireWireLink, 0);
OSMetaClassDefineReservedUnused(IOFireWireLink, 1);
OSMetaClassDefineReservedUnused(IOFireWireLink, 2);
OSMetaClassDefineReservedUnused(IOFireWireLink, 3);
OSMetaClassDefineReservedUnused(IOFireWireLink, 4);
OSMetaClassDefineReservedUnused(IOFireWireLink, 5);
OSMetaClassDefineReservedUnused(IOFireWireLink, 6);
OSMetaClassDefineReservedUnused(IOFireWireLink, 7);
OSMetaClassDefineReservedUnused(IOFireWireLink, 8);

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

IOFireWireController * IOFireWireLink::createController()
{
    IOFireWireController *control;

    control = OSTypeAlloc( IOFireWireController );
    if(NULL == control)
        return NULL;

    if(!control->init(this)) {
        control->release();
        control = NULL;
    }
    return control;
}

IOFWWorkLoop * IOFireWireLink::createWorkLoop()
{
    return IOFWWorkLoop::workLoop();
}

IOFireWireDevice *
IOFireWireLink::createDeviceNub(CSRNodeUniqueID guid, const IOFWNodeScan *scan)
{
    IOFireWireDevice *newDevice;
    OSDictionary *propTable;

    newDevice = OSTypeAlloc( IOFireWireDevice );

    if (!newDevice)
        return NULL;

    do {
        OSObject * prop;
        propTable = OSDictionary::withCapacity(6);
        if (!propTable)
            continue;

        prop = OSNumber::withNumber(guid, 64);
        if(prop) {
            propTable->setObject(gFireWire_GUID, prop);
            prop->release();
        }
		prop = OSNumber::withNumber((OSSwapBigToHostInt32(scan->fSelfIDs[0]) & kFWSelfID0SP) >> kFWSelfID0SPPhase, 32);
		if(prop) {
            propTable->setObject(gFireWireSpeed, prop);
            prop->release();
        }

        if(!newDevice->init(propTable, scan)) {
            newDevice->release();
            newDevice = NULL;
        }
        
//        IOLog("IOFireWireLink::createDeviceNub - GUID is 0x%llx\n", guid );

        if( newDevice )
        {
            // use quadlet reads for config rom
            newDevice->setMaxPackLog(false, true, 2);
        }
    } while (false);
    if(propTable)
        propTable->release();	// done with it after init

    return newDevice;
}

IOFireWireController * IOFireWireLink::getController() const
{
    return fControl;
}

IOWorkLoop * IOFireWireLink::getWorkLoop() const
{
    return fWorkLoop;
}

IOFWWorkLoop * IOFireWireLink::getFireWireWorkLoop() const
{
    return fWorkLoop;
}

IOFWDCLPool *
IOFireWireLink::createDCLPool ( 
	UInt32				capacity )
{
	return NULL ;
}

void IOFireWireLink::disablePHYPortOnSleep( UInt32 mask )
{
	// nothing to do
}

UInt32 * IOFireWireLink::getPingTimes ()
{
	return NULL ;
}

IOReturn IOFireWireLink::handleAsyncCompletion( IOFWCommand *cmd, IOReturn status )
{
	// nothing to do
	
	return kIOReturnSuccess;
}

void IOFireWireLink::handleSystemShutDown( UInt32 messageType )
{
	// nothing to do
}

void IOFireWireLink::configureAsyncRobustness( bool enabled )
{
	// nothing to do
}

bool IOFireWireLink::isPhysicalAccessEnabledForNodeID( UInt16 nodeID )
{
	return false;
}

void IOFireWireLink::notifyInvalidSelfIDs (void)
{

}

IOReturn IOFireWireLink::asyncPHYPacket( UInt32 data, UInt32 data2, IOFWAsyncPHYCommand * cmd )
{
	return kIOReturnUnsupported;
}

bool IOFireWireLink::enterLoggingMode( void )
{
	return false;
}

IOReturn IOFireWireLink::getCycleTimeAndUpTime( UInt32 &cycleTime, UInt64 &uptime )
{
	return kIOReturnUnsupported;
}

UInt32 IOFireWireLink::setLinkMode( UInt32 arg1, UInt32 arg2 )
{
	return 0;
}
