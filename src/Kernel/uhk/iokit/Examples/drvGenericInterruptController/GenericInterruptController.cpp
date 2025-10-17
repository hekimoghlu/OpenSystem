/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 20, 2024.
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
 */

#include <IOKit/IOPlatformExpert.h>

#include "GenericInterruptController.h"

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#undef  super
#define super IOInterruptController

IODefineMetaClassAndStructors(GenericInterruptController,
    IOInterruptController);

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */


bool
GenericInterruptController::start(IOService *provider)
{
	IOInterruptAction    handler;
	IOSymbol             *interruptControllerName;

	// If needed call the parents start.
	if (!super::start(provider)) {
		return false;
	}

	// Map the device's memory and initalize its state.

	// For now you must allocate storage for the vectors.
	// This will probably changed to something like: initVectors(numVectors).
	// In the mean time something like this works well.
#if 0
	// Allocate the memory for the vectors.
	vectors = (IOInterruptVector *)IOMalloc(numVectors *
	    sizeof(IOInterruptVector));
	if (vectors == NULL) {
		return false;
	}
	bzero(vectors, numVectors * sizeof(IOInterruptVector));

	// Allocate locks for the vectors.
	for (cnt = 0; cnt < numVectors; cnt++) {
		vectors[cnt].interruptLock = IOLockAlloc();
		if (vectors[cnt].interruptLock == NULL) {
			for (cnt = 0; cnt < numVectors; cnt++) {
				if (vectors[cnt].interruptLock != NULL) {
					IOLockFree(vectors[cnt].interruptLock);
				}
			}
		}
	}
#endif

	// If you know that this interrupt controller is the primary
	// interrupt controller, use this to set it nub properties properly.
	// This may be done by the nub's creator.
	getPlatform()->setCPUInterruptProperties(provider);

	// register the interrupt handler so it can receive interrupts.
	handler = getInterruptHandlerAddress();
	provider->registerInterrupt(0, this, handler, 0);

	// Just like any interrupt source, you must enable it to receive interrupts.
	provider->enableInterrupt(0);

	// Set interruptControllerName to the proper symbol.
	//interruptControllerName = xxx;

	// Register this interrupt controller so clients can find it.
	getPlatform()->registerInterruptController(interruptControllerName, this);

	// All done, so return true.
	return true;
}

IOReturn
GenericInterruptController::getInterruptType(IOService *nub,
    int source,
    int *interruptType)
{
	if (interruptType == 0) {
		return kIOReturnBadArgument;
	}

	// Given the nub and source, set interruptType to level or edge.

	return kIOReturnSuccess;
}

// Sadly this just has to be replicated in every interrupt controller.
IOInterruptAction
GenericInterruptController::getInterruptHandlerAddress(void)
{
	return (IOInterruptAction)handleInterrupt;
}

// Handle all current interrupts.
IOReturn
GenericInterruptController::handleInterrupt(void * refCon,
    IOService * nub,
    int source)
{
	IOInterruptVector *vector;
	int               vectorNumber;

	while (1) {
		// Get vectorNumber from hardware some how and clear the event.

		// Break if there are no more vectors to handle.
		if (vectorNumber == 0 /*kNoVector*/) {
			break;
		}

		// Get the vector's date from the controller's array.
		vector = &vectors[vectorNumber];

		// Set the vector as active. This store must compleat before
		// moving on to prevent the disableInterrupt fuction from
		// geting out of sync.
		vector->interruptActive = 1;
		//sync();
		//isync();

		// If the vector is not disabled soft, handle it.
		if (!vector->interruptDisabledSoft) {
			// Prevent speculative exacution as needed on your processor.
			//isync();

			// Call the handler if it exists.
			if (vector->interruptRegistered) {
				vector->handler(vector->target, vector->refCon,
				    vector->nub, vector->source);
			}
		} else {
			// Hard disable the vector if is was only soft disabled.
			vector->interruptDisabledHard = 1;
			disableVectorHard(vectorNumber, vector);
		}

		// Done with this vector so, set it back to inactive.
		vector->interruptActive = 0;
	}

	return kIOReturnSuccess;
}

bool
GenericInterruptController::vectorCanBeShared(long vectorNumber,
    IOInterruptVector *vector)
{
	// Given the vector number and the vector data, return if it can be shared.
	return true;
}

void
GenericInterruptController::initVector(long vectorNumber,
    IOInterruptVector *vector)
{
	// Given the vector number and the vector data,
	// get the hardware ready for the vector to generate interrupts.
	// Make sure the vector is left disabled.
}

void
GenericInterruptController::disableVectorHard(long vectorNumber,
    IOInterruptVector *vector)
{
	// Given the vector number and the vector data,
	// disable the vector at the hardware.
}

void
GenericInterruptController::enableVector(long vectorNumber,
    IOInterruptVector *vector)
{
	// Given the vector number and the vector data,
	// enable the vector at the hardware.
}

void
GenericInterruptController::causeVector(long vectorNumber,
    IOInterruptVector *vector)
{
	// Given the vector number and the vector data,
	// Set the vector pending and cause an interrupt at the parent controller.

	// cause the interrupt at the parent controller.  Source is usually zero,
	// but it could be different for your controller.
	getPlatform()->causeInterrupt(0);
}
