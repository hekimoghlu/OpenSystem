/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 2, 2023.
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
//#define IOASSERT 1	// Set to 1 to activate assert()

// public
#include <IOKit/firewire/IOFWCommand.h>
#include <IOKit/firewire/IOFireWireController.h>
#include <IOKit/firewire/IOFireWireNub.h>
#include <IOKit/firewire/IOLocalConfigDirectory.h>

// system
#include <IOKit/assert.h>
#include <IOKit/IOWorkLoop.h>
#include <IOKit/IOCommand.h>

OSDefineMetaClass( IOFWBusCommand, IOFWCommand )
OSDefineAbstractStructors(IOFWBusCommand, IOFWCommand)
OSMetaClassDefineReservedUnused(IOFWBusCommand, 0);

#pragma mark -

// initWithController
//
//

bool IOFWBusCommand::initWithController(IOFireWireController *control, FWBusCallback completion, void *refcon)
{
    if(!IOFWCommand::initWithController(control))
	return false;

    fSync = completion == NULL;
    fComplete = completion;
    fRefCon = refcon;
    return true;
}

// reinit
//
//

IOReturn IOFWBusCommand::reinit(FWBusCallback completion, void *refcon)
{
    if(fStatus == kIOReturnBusy || fStatus == kIOFireWirePending)
	return fStatus;

    fSync = completion == NULL;
    fComplete = completion;
    fRefCon = refcon;
    return kIOReturnSuccess;
}

// complete
//
//

IOReturn IOFWBusCommand::complete(IOReturn state)
{
    state = IOFWCommand::complete(state);
    if(fSync)
        fSyncWakeup->signal(state);
    else if(fComplete)
		(*fComplete)(fRefCon, state, fControl, this);
    return state;
}
