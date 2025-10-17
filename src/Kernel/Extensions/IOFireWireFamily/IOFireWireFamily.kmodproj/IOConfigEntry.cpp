/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 3, 2024.
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
#import <IOKit/firewire/IOLocalConfigDirectory.h>

// private
#import "FWDebugging.h"

// system
#import <libkern/c++/OSIterator.h>
#import <libkern/c++/OSData.h>
#import <libkern/c++/OSArray.h>

#import "IOConfigEntry.h"

OSDefineMetaClassAndStructors(IOConfigEntry, OSObject);

// free
//
//

void IOConfigEntry::free()
{
    if (fData) {
        fData->release();
        fData = NULL;
    }

    OSObject::free();
}

// create
//
//

IOConfigEntry* IOConfigEntry::create(UInt32 key, IOConfigKeyType type, OSObject *obj)
{
    IOConfigEntry* entry;
    entry = OSTypeAlloc( IOConfigEntry );
    if(!entry)
        return NULL;
    if(!entry->init()) {
        entry->release();
        return NULL;
    }
    assert(type == kConfigLeafKeyType ||
           type == kConfigDirectoryKeyType);
    entry->fKey = key;
    entry->fType = type;

	obj->retain() ;
    entry->fData = obj;
	
    return entry;
}

// create
//
//

IOConfigEntry* IOConfigEntry::create(UInt32 key, UInt32 value)
{
    IOConfigEntry* entry;

    if(value > kConfigEntryValue)
        return NULL;	// Too big to fit!!
    entry = OSTypeAlloc( IOConfigEntry );
    if(!entry)
        return NULL;
    if(!entry->init()) {
        entry->release();
        return NULL;
    }
    entry->fKey = key;
    entry->fType = kConfigImmediateKeyType;
    entry->fValue = value;
    return entry;
}

// create
//
//

IOConfigEntry* IOConfigEntry::create(UInt32 key, FWAddress address)
{
    IOConfigEntry* entry;
    entry = OSTypeAlloc( IOConfigEntry );
    if(!entry)
        return NULL;
    if(!entry->init()) {
        entry->release();
        return NULL;
    }
    entry->fKey = key;
    entry->fType = kConfigOffsetKeyType;
    entry->fAddr = address;
    return entry;
}

// totalSize
//
//

unsigned int IOConfigEntry::totalSize()
{
    unsigned int size = 0;
    switch(fType) {
	default:
            break;
        case kConfigLeafKeyType:
        {
            OSData *data = OSDynamicCast(OSData, fData);
            if(!data)
                return 0;
            // Round up size to multiple of 4, plus header
            size = (data->getLength()-1) / sizeof(UInt32) + 2;
            break;
            
        }
        case kConfigDirectoryKeyType:
        {
            IOLocalConfigDirectory *dir = OSDynamicCast(IOLocalConfigDirectory,
                                                            fData);
            if(!dir)
                return 0;	// Oops!
            const OSArray *entries = dir->getEntries();
            unsigned int i;
            size = 1 + entries->getCount();
            for(i=0; i<entries->getCount(); i++) {
                IOConfigEntry *entry = OSDynamicCast(IOConfigEntry, entries->getObject(i));
                if(!entry)
                    return 0;	// Oops!
                size += entry->totalSize();
            }
            break;
        }
    }
    return size;
}
