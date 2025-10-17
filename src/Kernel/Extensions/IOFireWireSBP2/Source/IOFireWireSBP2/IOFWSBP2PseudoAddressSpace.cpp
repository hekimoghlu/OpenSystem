/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 24, 2021.
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
// public
#include <IOKit/firewire/IOFireWireUnit.h>
#include <IOKit/firewire/IOFireWireDevice.h>

// private
#include "IOFWSBP2PseudoAddressSpace.h"

OSDefineMetaClassAndStructors(IOFWSBP2PseudoAddressSpace, IOFWPseudoAddressSpace);

OSMetaClassDefineReservedUnused(IOFWSBP2PseudoAddressSpace, 0);
OSMetaClassDefineReservedUnused(IOFWSBP2PseudoAddressSpace, 1);
OSMetaClassDefineReservedUnused(IOFWSBP2PseudoAddressSpace, 2);
OSMetaClassDefineReservedUnused(IOFWSBP2PseudoAddressSpace, 3);
OSMetaClassDefineReservedUnused(IOFWSBP2PseudoAddressSpace, 4);
OSMetaClassDefineReservedUnused(IOFWSBP2PseudoAddressSpace, 5);
OSMetaClassDefineReservedUnused(IOFWSBP2PseudoAddressSpace, 6);
OSMetaClassDefineReservedUnused(IOFWSBP2PseudoAddressSpace, 7);
OSMetaClassDefineReservedUnused(IOFWSBP2PseudoAddressSpace, 8);
OSMetaClassDefineReservedUnused(IOFWSBP2PseudoAddressSpace, 9);

#pragma mark -

// setAddressLo
//
//

void IOFWSBP2PseudoAddressSpace::setAddressLo( UInt32 addressLo )
{
	fBase.addressLo = addressLo;
}

// simpleRead
//
//

IOFWSBP2PseudoAddressSpace * IOFWSBP2PseudoAddressSpace::simpleRead(	IOFireWireBus *	control,
																		FWAddress *		addr, 
																		UInt32 			len, 
																		const void *	data)
{
    IOFWSBP2PseudoAddressSpace * me = new IOFWSBP2PseudoAddressSpace;
    do 
	{
        if(!me)
            break;
        
		if(!me->initAll(control, addr, len, simpleReader, NULL, (void *)me)) 
		{
            me->release();
            me = NULL;
            break;
        }
        
		me->fDesc = IOMemoryDescriptor::withAddress((void *)data, len, kIODirectionOut);
        if(!me->fDesc) 
		{
            me->release();
            me = NULL;
        }
		
    } while(false);

    return me;
}

// simpleRW
//
//

IOFWSBP2PseudoAddressSpace * IOFWSBP2PseudoAddressSpace::simpleRW(	IOFireWireBus *	control,
																	FWAddress *		addr, 
																	UInt32 			len, 
																	void *			data )
{
    IOFWSBP2PseudoAddressSpace * me = new IOFWSBP2PseudoAddressSpace;
    do 
	{
        if(!me)
            break;
    
		if(!me->initAll(control, addr, len, simpleReader, simpleWriter, (void *)me)) 
		{
            me->release();
            me = NULL;
            break;
        }
        
		me->fDesc = IOMemoryDescriptor::withAddress(data, len, kIODirectionInOut);
        if(!me->fDesc) 
		{
            me->release();
            me = NULL;
        }
		
    } while(false);

    return me;
}

// createPseudoAddressSpace
//
//

IOFWSBP2PseudoAddressSpace * IOFWSBP2PseudoAddressSpace::createPseudoAddressSpace( 	IOFireWireBus * control,
																					IOFireWireUnit * unit,
																					FWAddress *		addr, 
																					UInt32 			len, 
																					FWReadCallback 	reader, 
																					FWWriteCallback	writer, 
																					void *			refcon )
{
 
    IOFWSBP2PseudoAddressSpace *	space = NULL;
    IOFireWireDevice * 				device = NULL;
	
	space = new IOFWSBP2PseudoAddressSpace;
    
	if( space != NULL )
	{
		if( !space->initAll( control, addr, len, reader, writer, refcon ) ) 
		{
			space->release();
			space = NULL;
		}
	}
	
 	if( space != NULL )
	{
		device = OSDynamicCast( IOFireWireDevice, unit->getProvider() );
		if( device == NULL )
		{
			space->release();
			space = NULL;
		}
	}
	
	if( space != NULL )
	{
		space->addTrustedNode( device );
	}
	
	return space;
	
}
