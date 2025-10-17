/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 16, 2022.
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
#ifndef __IONDRV__
#define __IONDRV__

#include <IOKit/IORegistryEntry.h>
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Woverloaded-virtual"
#include <IOKit/IOInterruptEventSource.h>
#pragma clang diagnostic pop

#include <IOKit/ndrvsupport/IOMacOSTypes.h>
#include <IOKit/ndrvsupport/IONDRVSupport.h>

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef kAAPLRegEntryIDKey
#define kAAPLRegEntryIDKey      "AAPL,RegEntryID"
#endif

#ifndef kAAPLDisableMSIKey
#define kAAPLDisableMSIKey      "AAPL,DisableMSI"
#endif

#define MAKE_REG_ENTRY(regEntryID, obj)                          	   \
        (regEntryID)->opaque[0] = (void *) (((uintptr_t)obj)  - ((uintptr_t)gIOFramebufferKey)); \
        (regEntryID)->opaque[1] = (void *) ~(((uintptr_t)obj) - ((uintptr_t)gIOFramebufferKey)); \
        (regEntryID)->opaque[2] = (void *) 0x53696d65;                  \
        (regEntryID)->opaque[3] = (void *) 0x52756c7a;

#define REG_ENTRY_TO_OBJ_RET(regEntryID, obj, ret)                      \
        uintptr_t __obj;												\
        if ((__obj = ((uintptr_t *)regEntryID)[0])  			        \
             != ~((uintptr_t *)regEntryID)[1])        	return (ret);	\
        obj = (IORegistryEntry *)(__obj + (uintptr_t)gIOFramebufferKey);

#define REG_ENTRY_TO_OBJ(regEntryID, obj) 	REG_ENTRY_TO_OBJ_RET((regEntryID), (obj), -2538)

#define REG_ENTRY_TO_PT(regEntryID,obj)                                 \
        IORegistryEntry * obj;                                          \
		REG_ENTRY_TO_OBJ_RET((regEntryID), (obj), -2538)

#define REG_ENTRY_TO_SERVICE(regEntryID, type, obj)                     \
        IORegistryEntry * regEntry;                                     \
        type            * obj;                                          \
		REG_ENTRY_TO_OBJ(regEntryID, regEntry);							\
        if (0 == (obj = OSDynamicCast(type, regEntry)))                 \
            return (-2542);

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

class IONDRV : public OSObject
{
    OSDeclareAbstractStructors(IONDRV)

public:
    virtual IOReturn getSymbol( const char * symbolName,
                                IOLogicalAddress * address ) = 0;

    virtual const char * driverName( void ) = 0;

    virtual IOReturn doDriverIO( UInt32 commandID, void * contents,
                                 UInt32 commandCode, UInt32 commandKind ) = 0;
};

#endif /* __IONDRV__ */

