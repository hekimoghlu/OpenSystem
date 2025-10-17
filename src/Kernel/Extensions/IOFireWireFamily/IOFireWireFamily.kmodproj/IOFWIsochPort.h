/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 20, 2025.
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
 * Copyright (c) 1999-2002 Apple Computer, Inc.  All rights reserved.
 *
 * IOFWIsochPort is an abstract object that represents hardware on the bus
 * (locally or remotely) that sends or receives isochronous packets.
 * Local ports are implemented by the local device driver,
 * Remote ports are implemented by the driver for the remote device.
 *
 * HISTORY
 *
 */


#ifndef _IOKIT_IOFWISOCHPORT_H
#define _IOKIT_IOFWISOCHPORT_H

#include <libkern/c++/OSObject.h>
#include <IOKit/firewire/IOFireWireFamilyCommon.h>

/*! @class IOFWIsochPort
*/

class IOFWIsochPort : public OSObject
{
    OSDeclareAbstractStructors(IOFWIsochPort)

public:
	// Return maximum speed and channels supported
	// (bit n set = chan n supported)
    virtual IOReturn getSupported(IOFWSpeed &maxSpeed, UInt64 &chanSupported) = 0;

	// Allocate hardware resources for port
    virtual IOReturn allocatePort(IOFWSpeed speed, UInt32 chan) = 0;
    virtual IOReturn releasePort() = 0;		// Free hardware resources
    virtual IOReturn start() = 0;		// Start port processing packets
    virtual IOReturn stop() = 0;		// Stop processing packets

private:
    OSMetaClassDeclareReservedUnused(IOFWIsochPort, 0);
    OSMetaClassDeclareReservedUnused(IOFWIsochPort, 1);

};

#endif /* ! _IOKIT_IOFWISOCHPORT_H */

