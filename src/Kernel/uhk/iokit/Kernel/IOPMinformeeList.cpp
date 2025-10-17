/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 22, 2024.
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
#include <IOKit/pwr_mgt/IOPM.h>
#include <IOKit/pwr_mgt/IOPMinformeeList.h>
#include <IOKit/pwr_mgt/IOPMinformee.h>

#define super OSObject
OSDefineMetaClassAndStructors(IOPMinformeeList, OSObject)

//*********************************************************************************
// init
//
//*********************************************************************************
void
IOPMinformeeList::initialize( void )
{
	firstItem = NULL;
	length = 0;
}

//******************************************************************************
// getSharedRecursiveLock
//
//******************************************************************************
IORecursiveLock *
IOPMinformeeList::getSharedRecursiveLock( void )
{
	static IORecursiveLock *sharedListLock = NULL;

	/* A running system could have 50-60+ instances of IOPMInformeeList.
	 * They'll share this lock, since list insertion and removal is relatively
	 * rare, and generally tied to major events like device discovery.
	 *
	 * getSharedRecursiveLock() is called from IOStartIOKit to initialize
	 * the sharedListLock before any IOPMinformeeLists are instantiated.
	 *
	 * The IOPMinformeeList class will be around for the lifetime of the system,
	 * we don't worry about freeing this lock.
	 */

	if (NULL == sharedListLock) {
		sharedListLock = IORecursiveLockAlloc();
	}
	return sharedListLock;
}

//*********************************************************************************
// appendNewInformee
//
//*********************************************************************************
IOPMinformee *
IOPMinformeeList::appendNewInformee( IOService * newObject )
{
	IOPMinformee * newInformee;

	if (!newObject) {
		return NULL;
	}

	newInformee = IOPMinformee::withObject(newObject);

	if (!newInformee) {
		return NULL;
	}

	if (IOPMNoErr == addToList(newInformee)) {
		return newInformee;
	} else {
		newInformee->release();
		return NULL;
	}
}


//*********************************************************************************
// addToList
// *OBSOLETE* do not call from outside of this file.
// Try appendNewInformee() instead
//*********************************************************************************
IOReturn
IOPMinformeeList::addToList( IOPMinformee * newInformee )
{
	IORecursiveLock *listLock = getSharedRecursiveLock();
	IOReturn        ret = kIOReturnError;

	if (!listLock) {
		return ret;
	}

	IORecursiveLockLock(listLock);

	// Is new object already in the list?
	if (findItem(newInformee->whatObject) != NULL) {
		// object is present; just exit
		goto unlock_and_exit;
	}

	// add it to the front of the list
	newInformee->nextInList = firstItem;
	firstItem = newInformee;
	length++;
	ret = IOPMNoErr;

unlock_and_exit:
	IORecursiveLockUnlock(listLock);
	return ret;
}


//*********************************************************************************
// removeFromList
//
// Find the item in the list, unlink it, and free it.
//*********************************************************************************

IOReturn
IOPMinformeeList::removeFromList( IOService * theItem )
{
	IOPMinformee * item = firstItem;
	IOPMinformee * temp;
	IORecursiveLock    *listLock = getSharedRecursiveLock();

	if (NULL == item) {
		return IOPMNoErr;
	}
	if (!listLock) {
		return kIOReturnError;
	}

	IORecursiveLockLock( listLock );

	if (item->whatObject == theItem) {
		firstItem = item->nextInList;
		length--;
		item->release();
		goto unlock_and_exit;
	}

	while (item->nextInList != NULL) {
		if (item->nextInList->whatObject == theItem) {
			temp = item->nextInList;
			item->nextInList = temp->nextInList;
			length--;
			temp->release();
			goto unlock_and_exit;
		}
		item = item->nextInList;
	}

unlock_and_exit:
	IORecursiveLockUnlock(listLock);
	return IOPMNoErr;
}


//*********************************************************************************
// firstInList
//
//*********************************************************************************

IOPMinformee *
IOPMinformeeList::firstInList( void )
{
	return firstItem;
}

//*********************************************************************************
// nextInList
//
//*********************************************************************************

IOPMinformee *
IOPMinformeeList::nextInList( IOPMinformee * currentItem )
{
	if (currentItem != NULL) {
		return currentItem->nextInList;
	}
	return NULL;
}

//*********************************************************************************
// numberOfItems
//
//*********************************************************************************

unsigned long
IOPMinformeeList::numberOfItems( void )
{
	return length;
}

//*********************************************************************************
// findItem
//
// Look through the list for the one which points to the object identified
// by the parameter.  Return a pointer to the list item or NULL.
//*********************************************************************************

IOPMinformee *
IOPMinformeeList::findItem( IOService * driverOrChild )
{
	IOPMinformee * nextObject;

	nextObject = firstInList();
	while (nextObject != NULL) {
		if (nextObject->whatObject == driverOrChild) {
			return nextObject;
		}
		nextObject = nextInList(nextObject);
	}
	return NULL;
}



//*********************************************************************************
// free
//
// Free all items in the list, and then free the list itself
//*********************************************************************************

void
IOPMinformeeList::free(void )
{
	IOPMinformee * next = firstItem;

	while (next != NULL) {
		firstItem = next->nextInList;
		length--;
		next->release();
		next = firstItem;
	}
	super::free();
}
