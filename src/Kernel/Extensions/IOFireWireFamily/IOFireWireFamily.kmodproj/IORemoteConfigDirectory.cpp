/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 10, 2023.
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
#import <IOKit/firewire/IORemoteConfigDirectory.h>
#import <IOKit/firewire/IOFireWireDevice.h>

// private
#import "FWDebugging.h"

// system
#import <libkern/c++/OSIterator.h>
#import <libkern/c++/OSData.h>

OSDefineMetaClassAndStructors(IORemoteConfigDirectory, IOConfigDirectory)
OSMetaClassDefineReservedUnused(IORemoteConfigDirectory, 0);
OSMetaClassDefineReservedUnused(IORemoteConfigDirectory, 1);
OSMetaClassDefineReservedUnused(IORemoteConfigDirectory, 2);

// initWithOwnerOffset
//
//

bool
IORemoteConfigDirectory::initWithOwnerOffset( IOFireWireROMCache *rom,
                         int start, int type)
{

    // Do this first so that init can load ROM
    fROM = rom;
    fROM->retain();

    if( !IOConfigDirectory::initWithOffset(start, type) ) 
	{
		fROM->release();
		fROM = NULL;
		
        return false;       
    }

    return true;
}

// free
//
//

void
IORemoteConfigDirectory::free()
{
    if(fROM)
        fROM->release();
    IOConfigDirectory::free();

}

// withOwnerOffset
//
//

IOConfigDirectory *
IORemoteConfigDirectory::withOwnerOffset( IOFireWireROMCache *rom,
                                           int start, int type)
{
    IORemoteConfigDirectory *dir;

    dir = OSTypeAlloc( IORemoteConfigDirectory );
    if( !dir )
        return NULL;

    if( !dir->initWithOwnerOffset(rom, start, type) ) 
	{
        dir->release();
        dir = NULL;
    }
    return dir;
}

// getBase
//
//

const UInt32 *IORemoteConfigDirectory::getBase()
{
    return ((const UInt32 *)fROM->getBytesNoCopy())+fStart+1;
}

// update
//
//

IOReturn IORemoteConfigDirectory::update(UInt32 offset, const UInt32 *&romBase)
{
	// unsupported
	
	return kIOReturnError;
}

IOConfigDirectory *
IORemoteConfigDirectory::getSubDir(int start, int type)
{
    return withOwnerOffset(fROM, start, type);
}

// lockData
//
//

const UInt32 * IORemoteConfigDirectory::lockData( void )
{
	fROM->lock();
	return (UInt32 *)fROM->getBytesNoCopy();
}

// unlockData
//
//

void IORemoteConfigDirectory::unlockData( void )
{
	fROM->unlock();
}

// updateROMCache
//
//

IOReturn IORemoteConfigDirectory::updateROMCache( UInt32 offset, UInt32 length )
{
	return fROM->updateROMCache( offset, length );
}

// checkROMState
//
//

IOReturn IORemoteConfigDirectory::checkROMState( void )
{
	return fROM->checkROMState();
}
