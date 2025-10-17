/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 1, 2025.
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
#include <IOKit/pwr_mgt/IOPMPowerSourceList.h>
#include <IOKit/pwr_mgt/IOPMPowerSource.h>

#define super OSObject
OSDefineMetaClassAndStructors(IOPMPowerSourceList, OSObject)

//******************************************************************************
// init
//
//******************************************************************************
void
IOPMPowerSourceList::initialize( void )
{
	firstItem = NULL;
	length = 0;
}

//******************************************************************************
// addToList
//
//******************************************************************************

IOReturn
IOPMPowerSourceList::addToList(IOPMPowerSource *newPowerSource)
{
	IOPMPowerSource * nextPowerSource;

	// Is new object already in the list?
	nextPowerSource = firstItem;
	while (nextPowerSource != NULL) {
		if (nextPowerSource == newPowerSource) {
			// yes, just return
			return IOPMNoErr;
		}
		nextPowerSource = nextInList(nextPowerSource);
	}

	// add it to list
	newPowerSource->nextInList = firstItem;
	firstItem = newPowerSource;
	length++;
	return IOPMNoErr;
}


//******************************************************************************
// firstInList
//
//******************************************************************************

IOPMPowerSource *
IOPMPowerSourceList::firstInList( void )
{
	return firstItem;
}

//******************************************************************************
// nextInList
//
//******************************************************************************

IOPMPowerSource *
IOPMPowerSourceList::nextInList(IOPMPowerSource *currentItem)
{
	if (currentItem != NULL) {
		return currentItem->nextInList;
	}
	return NULL;
}

//******************************************************************************
// numberOfItems
//
//******************************************************************************

unsigned long
IOPMPowerSourceList::numberOfItems( void )
{
	return length;
}

//******************************************************************************
// removeFromList
//
// Find the item in the list, unlink it, and free it.
//******************************************************************************

IOReturn
IOPMPowerSourceList::removeFromList( IOPMPowerSource * theItem )
{
	IOPMPowerSource * item = firstItem;
	IOPMPowerSource * temp;

	if (NULL == item) {
		goto exit;
	}

	if (item == theItem) {
		firstItem = item->nextInList;
		length--;
		item->release();
		return IOPMNoErr;
	}
	while (item->nextInList != NULL) {
		if (item->nextInList == theItem) {
			temp = item->nextInList;
			item->nextInList = temp->nextInList;
			length--;
			temp->release();
			return IOPMNoErr;
		}
		item = item->nextInList;
	}

exit:
	return IOPMNoErr;
}


//******************************************************************************
// free
//
// Free all items in the list, and then free the list itself
//******************************************************************************

void
IOPMPowerSourceList::free(void )
{
	IOPMPowerSource * next = firstItem;

	while (next != NULL) {
		firstItem = next->nextInList;
		length--;
		next->release();
		next = firstItem;
	}
	super::free();
}
