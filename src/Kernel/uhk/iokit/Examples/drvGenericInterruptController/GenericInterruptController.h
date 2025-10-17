/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 19, 2022.
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

#ifndef _IOKIT_GENERICINTERRUPTCONTROLLER_H
#define _IOKIT_GENERICINTERRUPTCONTROLLER_H

#include <IOKit/IOInterrupts.h>
#include <IOKit/IOInterruptController.h>

class GenericInterruptController : public IOInterruptController
{
	IODeclareDefaultStructors(GenericInterruptController);

public:
// There should be a method to start or init the controller.
// Its nature is up to you.
	virtual bool start(IOService *provider);

// Returns the type of a vector: level or edge.  This will probably get
// replaced but a default method and a new method getVectorType.
	virtual IOReturn getInterruptType(IOService *nub, int source,
	    int *interruptType);

// Returns a function pointer for the interrupt handler.
// Sadly, egcs prevents this from being done by the base class.
	virtual IOInterruptAction getInterruptHandlerAddress(void);

// The actual interrupt handler.
	virtual IOReturn handleInterrupt(void *refCon,
	    IOService *nub, int source);


// Should return true if this vector can be shared.
// The base class return false, so this method only need to be implemented
// if the controller needs to support shared interrupts.
// No other work is required to support shared interrupts.
	virtual bool vectorCanBeShared(long vectorNumber, IOInterruptVector *vector);

// Do any hardware initalization for this vector.  Leave the vector
// hard disabled.
	virtual void initVector(long vectorNumber, IOInterruptVector *vector);

// Disable this vector at the hardware.
	virtual void disableVectorHard(long vectorNumber, IOInterruptVector *vector);

// Enable this vector at the hardware.
	virtual void enableVector(long vectorNumber, IOInterruptVector *vector);

// Cause an interrupt on this vector.
	virtual void causeVector(long vectorNumber, IOInterruptVector *vector);
};

#endif /* ! _IOKIT_GENERICINTERRUPTCONTROLLER_H */
