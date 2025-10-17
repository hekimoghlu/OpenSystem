/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 11, 2022.
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
#ifndef _IOKIT_IOHIDIUNKNOWN_H
#define _IOKIT_IOHIDIUNKNOWN_H

#include <IOKit/IOCFPlugIn.h>

__BEGIN_DECLS
//extern void *IOHIDLibFactory(CFAllocatorRef allocator, CFUUIDRef typeID);
__END_DECLS

class IOHIDIUnknown {

public:
    struct InterfaceMap {
        IUnknownVTbl *pseudoVTable;
        IOHIDIUnknown *obj;
    };

private:
    IOHIDIUnknown(IOHIDIUnknown &src);	// Disable copy constructor
    void operator =(IOHIDIUnknown &src);
    IOHIDIUnknown() : refCount(1) { };

protected:

    static int factoryRefCount;
    static void factoryAddRef();
    static void factoryRelease();

    IOHIDIUnknown(void *unknownVTable);
    virtual ~IOHIDIUnknown(); // Also virtualise destructor

    static HRESULT genericQueryInterface(void *self, REFIID iid, void **ppv);
    static UInt32 genericAddRef(void *self);
    static UInt32 genericRelease(void *self);

protected:

    UInt32 refCount;
    InterfaceMap iunknown;

public:
    virtual HRESULT queryInterface(REFIID iid, void **ppv) = 0;
    virtual UInt32 addRef();
    virtual UInt32 release();
};

#endif /* !_IOKIT_IOHIDIUNKNOWN_H */
