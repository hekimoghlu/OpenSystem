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
#include <TargetConditionals.h>

#include <IOKit/hid/IOHIDDevicePlugIn.h>
#include <IOKit/hid/IOHIDServicePlugIn.h>
#include "IOHIDIUnknown.h"
#include <stdatomic.h>
#include <os/log.h>

int IOHIDIUnknown::factoryRefCount = 0;


void IOHIDIUnknown::factoryAddRef()
{
    CFUUIDRef factoryId = kIOHIDDeviceFactoryID;
    CFPlugInAddInstanceForFactory(factoryId);
}

void IOHIDIUnknown::factoryRelease()
{
    CFUUIDRef factoryId = kIOHIDDeviceFactoryID;
    CFPlugInRemoveInstanceForFactory(factoryId);

}

IOHIDIUnknown::IOHIDIUnknown(void *unknownVTable)
: refCount(1)
{
    iunknown.pseudoVTable = (IUnknownVTbl *) unknownVTable;
    iunknown.obj = this;

    factoryAddRef();
};

IOHIDIUnknown::~IOHIDIUnknown()
{
    factoryRelease();
}

UInt32 IOHIDIUnknown::addRef()
{
    return atomic_fetch_add((_Atomic UInt32*)&refCount, 1) + 1;
}

UInt32 IOHIDIUnknown::release()
{
    UInt32 retVal = atomic_fetch_sub((_Atomic UInt32*)&refCount, 1);
    
    if (retVal < 1) {
        os_log_fault(OS_LOG_DEFAULT, "Over Release IOHIDIUnknown Reference");
    } else if (retVal == 1) {
        delete this;
    }
    
    return retVal - 1;
}

HRESULT IOHIDIUnknown::
genericQueryInterface(void *self, REFIID iid, void **ppv)
{
    IOHIDIUnknown *me = ((InterfaceMap *) self)->obj;
    return me->queryInterface(iid, ppv);
}

UInt32 IOHIDIUnknown::genericAddRef(void *self)
{
    IOHIDIUnknown *me = ((InterfaceMap *) self)->obj;
    return me->addRef();
}

UInt32 IOHIDIUnknown::genericRelease(void *self)
{
    IOHIDIUnknown *me = ((InterfaceMap *) self)->obj;
    return me->release();
}
