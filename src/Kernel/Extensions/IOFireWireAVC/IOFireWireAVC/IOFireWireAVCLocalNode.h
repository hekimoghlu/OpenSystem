/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 12, 2024.
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
 *
 *	IOFireWireAVCLocalNode.h
 *
 * Class to initialize the Local node's AVC Target mode support
 *
 */

#include <IOKit/IOService.h>
#include <IOKit/IOLib.h>
#include <IOKit/IOMessage.h>

#include <IOKit/firewire/IOFireWireNub.h>
#include <IOKit/firewire/IOFireWireBus.h>
#include <IOKit/firewire/IOFWAddressSpace.h>

#include <IOKit/avc/IOFireWirePCRSpace.h>
#include <IOKit/avc/IOFireWireAVCTargetSpace.h>

class IOFireWireAVCLocalNode : public IOService
{
    OSDeclareDefaultStructors(IOFireWireAVCLocalNode)

protected:
	bool						fStarted;	
    IOFireWireNub *				fDevice;
    IOFireWirePCRSpace *		fPCRSpace;
    IOFireWireAVCTargetSpace *	fAVCTargetSpace;
	
public:

	// IOService overrides
    virtual bool		start(IOService *provider);
	virtual void		stop(IOService *provider);
	virtual void		free();
    virtual bool		finalize(IOOptionBits options);
    virtual IOReturn	message(UInt32 type, IOService *provider, void *argument);
};
	