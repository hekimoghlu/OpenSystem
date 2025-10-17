/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 9, 2025.
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
#ifndef _IOKIT_IOCFPLUGIN_H_
#define _IOKIT_IOCFPLUGIN_H_

/* IOCFPlugIn.h
 */
#include <sys/cdefs.h>

__BEGIN_DECLS

#include <CoreFoundation/CFPlugIn.h>
#if COREFOUNDATION_CFPLUGINCOM_SEPARATE
#include <CoreFoundation/CFPlugInCOM.h>
#endif

#include <IOKit/IOKitLib.h>

/* C244E858-109C-11D4-91D4-0050E4C6426F */
#define kIOCFPlugInInterfaceID CFUUIDGetConstantUUIDWithBytes(NULL,	\
    0xC2, 0x44, 0xE8, 0x58, 0x10, 0x9C, 0x11, 0xD4,			\
    0x91, 0xD4, 0x00, 0x50, 0xE4, 0xC6, 0x42, 0x6F)


#define IOCFPLUGINBASE							\
    UInt16	version;						\
    UInt16	revision;						\
    IOReturn (*Probe)(void *thisPointer, CFDictionaryRef propertyTable,	\
                    io_service_t service, SInt32 * order);		\
    IOReturn (*Start)(void *thisPointer, CFDictionaryRef propertyTable,	\
                      io_service_t service);				\
    IOReturn (*Stop)(void *thisPointer)

typedef struct IOCFPlugInInterfaceStruct {
    IUNKNOWN_C_GUTS;
    IOCFPLUGINBASE;
} IOCFPlugInInterface;


kern_return_t
IOCreatePlugInInterfaceForService(io_service_t service,
                CFUUIDRef pluginType, CFUUIDRef interfaceType,
                IOCFPlugInInterface *** theInterface, SInt32 * theScore);

kern_return_t
IODestroyPlugInInterface(IOCFPlugInInterface ** interface);

__END_DECLS

#endif /* !_IOKIT_IOCFPLUGIN_H_ */
