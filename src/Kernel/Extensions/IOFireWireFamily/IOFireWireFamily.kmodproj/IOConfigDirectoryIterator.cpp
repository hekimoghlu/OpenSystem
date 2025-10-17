/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 16, 2023.
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
#import <IOKit/firewire/IOConfigDirectory.h>
#import <IOKit/firewire/IOFireWireDevice.h>

// private
#import "FWDebugging.h"
#import "IOConfigDirectoryIterator.h"

// system
#import <libkern/c++/OSIterator.h>

OSDefineMetaClassAndStructors(IOConfigDirectoryIterator, OSIterator)

// init
//
//

IOReturn IOConfigDirectoryIterator::init(IOConfigDirectory *owner,
                                  		 UInt32 testVal, UInt32 testMask)
{
	IOReturn status = kIOReturnSuccess;
	
    if( !OSIterator::init() )
        status = kIOReturnError;
	
	if( status == kIOReturnSuccess )
	{
		fDirectorySet = OSSet::withCapacity(2);
		if( fDirectorySet == NULL )
			status = kIOReturnNoMemory;
	}
	
	int position = 0;
	while( status == kIOReturnSuccess && position < owner->getNumEntries() ) 
	{
		UInt32 value;
		IOConfigDirectory * next;
		
		status = owner->getIndexEntry( position, value );
		if( status == kIOReturnSuccess && (value & testMask) == testVal ) 
		{
			status = owner->getIndexValue( position, next );
			if( status == kIOReturnSuccess )
			{
				fDirectorySet->setObject( next );
				next->release();
			}
		}
		
		position++;
	}
    
	if( status == kIOReturnSuccess )
	{
		fDirectoryIterator = OSCollectionIterator::withCollection( fDirectorySet );
		if( fDirectoryIterator == NULL )
			status = kIOReturnNoMemory;
	}
	
    return status;
}

// free
//
//

void IOConfigDirectoryIterator::free()
{
	if( fDirectoryIterator != NULL )
	{
		fDirectoryIterator->release();
		fDirectoryIterator = NULL;
	}
	
	if( fDirectorySet != NULL )
	{
		fDirectorySet->release();
		fDirectorySet = NULL;
	}
		
    OSIterator::free();
}

// reset
//
//

void IOConfigDirectoryIterator::reset()
{
    fDirectoryIterator->reset();
}

// isValid
//
//

bool IOConfigDirectoryIterator::isValid()
{
    return true;
}

// getNextObject
//
//

OSObject *IOConfigDirectoryIterator::getNextObject()
{
	return fDirectoryIterator->getNextObject();
}
