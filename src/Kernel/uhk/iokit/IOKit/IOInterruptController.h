/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 12, 2023.
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
 * Copyright (c) 1999 Apple Computer, Inc.  All rights reserved.
 *
 *  DRI: Josh de Cesare
 *
 */


#ifndef _IOKIT_IOINTERRUPTCONTROLLER_H
#define _IOKIT_IOINTERRUPTCONTROLLER_H

#include <IOKit/IOLocks.h>
#include <IOKit/IOService.h>
#include <IOKit/IOInterrupts.h>


class IOSharedInterruptController;

struct IOInterruptVector {
	volatile char               interruptActive;
	volatile char               interruptDisabledSoft;
	volatile char               interruptDisabledHard;
	volatile char               interruptRegistered;
	IOLock *                    interruptLock;
	IOService *                 nub;
	int                         source;
	void *                      target;
	IOInterruptHandler          handler;
	void *                      refCon;
	IOSharedInterruptController *sharedController;
};

typedef struct IOInterruptVector IOInterruptVector;

#if __LP64__
typedef int32_t IOInterruptVectorNumber;
#else
typedef long IOInterruptVectorNumber;
#endif

class IOInterruptController : public IOService
{
	OSDeclareAbstractStructors(IOInterruptController);

protected:
	IOInterruptVector *vectors;
	IOSimpleLock      *controllerLock;

	struct ExpansionData { };
	ExpansionData *ioic_reserved;

public:
	virtual IOReturn registerInterrupt(IOService *nub, int source,
	    void *target,
	    IOInterruptHandler handler,
	    void *refCon);
	virtual IOReturn unregisterInterrupt(IOService *nub, int source);

	virtual IOReturn getInterruptType(IOService *nub, int source,
	    int *interruptType);

	virtual IOReturn enableInterrupt(IOService *nub, int source);
	virtual IOReturn disableInterrupt(IOService *nub, int source);
	virtual IOReturn causeInterrupt(IOService *nub, int source);

	virtual IOInterruptAction getInterruptHandlerAddress(void);
	virtual IOReturn handleInterrupt(void *refCon, IOService *nub,
	    int source);

// Methods to be overridden for simplifed interrupt controller subclasses.

	virtual bool vectorCanBeShared(IOInterruptVectorNumber vectorNumber, IOInterruptVector *vector);
	virtual void initVector(IOInterruptVectorNumber vectorNumber, IOInterruptVector *vector);
	virtual int  getVectorType(IOInterruptVectorNumber vectorNumber, IOInterruptVector *vector);
	virtual void disableVectorHard(IOInterruptVectorNumber vectorNumber, IOInterruptVector *vector);
	virtual void enableVector(IOInterruptVectorNumber vectorNumber, IOInterruptVector *vector);
	virtual void causeVector(IOInterruptVectorNumber vectorNumber, IOInterruptVector *vector);
	virtual void setCPUInterruptProperties(IOService *service);

	virtual void sendIPI(unsigned int cpu_id, bool deferred);
	virtual void cancelDeferredIPI(unsigned int cpu_id);

	OSMetaClassDeclareReservedUsedX86(IOInterruptController, 0);
	OSMetaClassDeclareReservedUsedX86(IOInterruptController, 1);
	OSMetaClassDeclareReservedUsedX86(IOInterruptController, 2);
	OSMetaClassDeclareReservedUnused(IOInterruptController, 3);
	OSMetaClassDeclareReservedUnused(IOInterruptController, 4);
	OSMetaClassDeclareReservedUnused(IOInterruptController, 5);

public:
// Generic methods (not to be overriden).

	void timeStampSpuriousInterrupt(void);
	void timeStampInterruptHandlerStart(IOInterruptVectorNumber vectorNumber, IOInterruptVector *vector);
	void timeStampInterruptHandlerEnd(IOInterruptVectorNumber vectorNumber, IOInterruptVector *vector);

private:
	void timeStampInterruptHandlerInternal(bool isStart, IOInterruptVectorNumber vectorNumber, IOInterruptVector *vector);
};


class IOSharedInterruptController : public IOInterruptController
{
	OSDeclareDefaultStructors(IOSharedInterruptController);

private:
	IOService         *provider;
	int               numVectors;
	int               vectorsRegistered;
	int               vectorsEnabled;
	volatile int      controllerDisabled;
	bool              sourceIsLevel;

	struct ExpansionData { };
	ExpansionData *iosic_reserved __unused;

public:
	virtual IOReturn initInterruptController(IOInterruptController *parentController, OSData *parentSource);

	virtual IOReturn registerInterrupt(IOService *nub, int source,
	    void *target,
	    IOInterruptHandler handler,
	    void *refCon) APPLE_KEXT_OVERRIDE;
	virtual IOReturn unregisterInterrupt(IOService *nub, int source) APPLE_KEXT_OVERRIDE;

	virtual IOReturn getInterruptType(IOService *nub, int source,
	    int *interruptType) APPLE_KEXT_OVERRIDE;

	virtual IOReturn enableInterrupt(IOService *nub, int source) APPLE_KEXT_OVERRIDE;
	virtual IOReturn disableInterrupt(IOService *nub, int source) APPLE_KEXT_OVERRIDE;

	virtual IOInterruptAction getInterruptHandlerAddress(void) APPLE_KEXT_OVERRIDE;
	virtual IOReturn handleInterrupt(void *refCon, IOService *nub, int source) APPLE_KEXT_OVERRIDE;

	OSMetaClassDeclareReservedUnused(IOSharedInterruptController, 0);
	OSMetaClassDeclareReservedUnused(IOSharedInterruptController, 1);
	OSMetaClassDeclareReservedUnused(IOSharedInterruptController, 2);
	OSMetaClassDeclareReservedUnused(IOSharedInterruptController, 3);
};


#endif /* ! _IOKIT_IOINTERRUPTCONTROLLER_H */
