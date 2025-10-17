/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 23, 2022.
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
/* IOArray.h created by rsulack on Thu 11-Sep-1997 */

#define IOKIT_ENABLE_SHARED_PTR

#include <libkern/OSDebug.h>

#include <libkern/c++/OSCollection.h>
#include <libkern/c++/OSDictionary.h>

#include <IOKit/IOKitDebug.h>

#define super OSObject

OSDefineMetaClassAndAbstractStructors(OSCollection, OSObject)


OSMetaClassDefineReservedUsedX86(OSCollection, 0);
OSMetaClassDefineReservedUsedX86(OSCollection, 1);
OSMetaClassDefineReservedUnused(OSCollection, 2);
OSMetaClassDefineReservedUnused(OSCollection, 3);
OSMetaClassDefineReservedUnused(OSCollection, 4);
OSMetaClassDefineReservedUnused(OSCollection, 5);
OSMetaClassDefineReservedUnused(OSCollection, 6);
OSMetaClassDefineReservedUnused(OSCollection, 7);

bool
OSCollection::init()
{
	if (!super::init()) {
		return false;
	}

	updateStamp = 0;

	return true;
}

void
OSCollection::haveUpdated()
{
	if (fOptions & kImmutable) {
		if (!(gIOKitDebug & kOSRegistryModsMode)) {
			panic("Trying to change a collection in the registry");
		} else {
			OSReportWithBacktrace("Trying to change a collection in the registry");
		}
	}
	updateStamp++;
}

unsigned
OSCollection::setOptions(unsigned options, unsigned mask, void *)
{
	unsigned old = fOptions;

	if (mask) {
		fOptions = (old & ~mask) | (options & mask);
	}

	return old;
}

OSSharedPtr<OSCollection>
OSCollection::copyCollection(OSDictionary *cycleDict)
{
	if (cycleDict) {
		OSObject *obj = cycleDict->getObject((const OSSymbol *) this);

		return OSSharedPtr<OSCollection>(reinterpret_cast<OSCollection *>(obj), OSRetain);
	} else {
		// If we are here it means that there is a collection subclass that
		// hasn't overridden the copyCollection method.  In which case just
		// return a reference to ourselves.
		// Hopefully this collection will not be inserted into the registry
		return OSSharedPtr<OSCollection>(this, OSRetain);
	}
}

bool
OSCollection::iterateObjects(void * refcon, bool (*callback)(void * refcon, OSObject * object))
{
	uint64_t     iteratorStore[2];
	unsigned int initialUpdateStamp;
	bool         done;

	assert(iteratorSize() < sizeof(iteratorStore));

	if (!initIterator(&iteratorStore[0])) {
		return false;
	}

	initialUpdateStamp = updateStamp;
	done = false;
	do{
		OSObject * object;
		if (!getNextObjectForIterator(&iteratorStore[0], &object)) {
			break;
		}
		done = callback(refcon, object);
	}while (!done && (initialUpdateStamp == updateStamp));

	return initialUpdateStamp == updateStamp;
}

static bool
OSCollectionIterateObjectsBlock(void * refcon, OSObject * object)
{
	bool (^block)(OSObject * object) = (typeof(block))refcon;
	return block(object);
}

bool
OSCollection::iterateObjects(bool (^block)(OSObject * object))
{
	return iterateObjects((void *) block, OSCollectionIterateObjectsBlock);
}
