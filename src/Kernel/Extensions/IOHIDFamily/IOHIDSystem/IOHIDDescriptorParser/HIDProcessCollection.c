/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 26, 2022.
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
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 28, 2023.
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
 */#include "HIDLib.h"

/*
 *------------------------------------------------------------------------------
 *
 * HIDProcessCollection - Process a Collection MainItem
 *
 *	 Input:
 *			  ptDescriptor			- The Descriptor Structure
 *			  ptPreparsedData		- The PreParsedData Structure
 *	 Output:
 *			  ptDescriptor			- The Descriptor Structure
 *			  ptPreparsedData		- The PreParsedData Structure
 *	 Returns:
 *			  kHIDSuccess		   - Success
 *			  kHIDNullPointerErr	  - Argument, Pointer was Null
 *
 *------------------------------------------------------------------------------
*/
OSStatus HIDProcessCollection(HIDReportDescriptor *ptDescriptor, HIDPreparsedDataPtr ptPreparsedData)
{
	HIDCollection *collections;
	HIDCollection *ptCollection;
	int parent;
	int iCollection;
/*
 *	Disallow NULL Pointers
*/
	if ((ptDescriptor == NULL) || (ptPreparsedData == NULL))
		return kHIDNullPointerErr;
/*
 *	Initialize the new Collection Structure
*/
	iCollection = ptPreparsedData->collectionCount++;
	collections = ptPreparsedData->collections;
	ptCollection = &collections[iCollection];
	ptCollection->data = ptDescriptor->item.unsignedValue;
	ptCollection->firstUsageItem = ptDescriptor->firstUsageItem;
	ptCollection->usageItemCount = ptPreparsedData->usageItemCount - ptDescriptor->firstUsageItem;
	ptDescriptor->firstUsageItem = ptPreparsedData->usageItemCount;
	ptCollection->children = 0;
	ptCollection->nextSibling = ptDescriptor->sibling;
	ptDescriptor->sibling = 0;
	ptCollection->firstChild = 0;
	ptCollection->usagePage = ptDescriptor->globals.usagePage;
	ptCollection->firstReportItem = ptPreparsedData->reportItemCount;
/*
 *	Set up the relationship with the parent Collection
*/
	parent = ptDescriptor->parent;
	ptCollection->parent = parent;
	collections[parent].firstChild = iCollection;
	collections[parent].children++;
	ptDescriptor->parent = iCollection;
/*
 *	Save the parent Collection Information on the stack
*/
	ptDescriptor->collectionStack[ptDescriptor->collectionNesting++] = parent;
	return kHIDSuccess;
}

/*
 *------------------------------------------------------------------------------
 *
 * HIDProcessEndCollection - Process an EndCollection MainItem
 *
 *	 Input:
 *			  ptDescriptor			- The Descriptor Structure
 *			  ptPreparsedData		- The PreParsedData Structure
 *	 Output:
 *			  ptPreparsedData		- The PreParsedData Structure
 *	 Returns:
 *			  kHIDSuccess		   - Success
 *			  kHIDNullPointerErr	  - Argument, Pointer was Null
 *
 *------------------------------------------------------------------------------
*/
OSStatus HIDProcessEndCollection(HIDReportDescriptor *ptDescriptor, HIDPreparsedDataPtr ptPreparsedData)
{
	HIDCollection *ptCollection;
	int iCollection;
/*
 *	Remember the number of ReportItem MainItems in this Collection
*/
	ptCollection = &ptPreparsedData->collections[ptDescriptor->parent];
	ptCollection->reportItemCount = ptPreparsedData->reportItemCount - ptCollection->firstReportItem;
/*
 *	Restore the parent Collection Data
*/
	iCollection = ptDescriptor->collectionStack[--ptDescriptor->collectionNesting];
	ptDescriptor->sibling = ptDescriptor->parent;
	ptDescriptor->parent = iCollection;
	return kHIDSuccess;
}

