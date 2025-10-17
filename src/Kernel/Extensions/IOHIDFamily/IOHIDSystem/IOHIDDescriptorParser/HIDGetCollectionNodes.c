/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 10, 2025.
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
 * Date: Monday, May 8, 2023.
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
 */#include "IOHIDDescriptorParser.h"
#include "IOHIDDescriptorParserPrivate.h"

/*
 *------------------------------------------------------------------------------
 *
 * HIDGetCollectionNodes - Get the Collections Database
 *
 *	 Input:
 *			  ptLinkCollectionNodes		  - Node Array provided by caller
 *			  piLinkCollectionNodesLength - Maximum Nodes
 *	 Output:
 *			  piLinkCollectionNodesLength - Actual number of Nodes
 *	 Returns:
 *			  kHIDSuccess		  - Success
 *			  kHIDNullPointerErr	 - Argument, Pointer was Null
 *			  HidP_NotEnoughRoom   - More Nodes than space for them
 *
 *------------------------------------------------------------------------------
*/
OSStatus HIDGetCollectionNodes(HIDCollectionNodePtr ptLinkCollectionNodes,
										UInt32 *piLinkCollectionNodesLength,
										HIDPreparsedDataRef preparsedDataRef)
{
	HIDPreparsedDataPtr ptPreparsedData = (HIDPreparsedDataPtr) preparsedDataRef;
	HIDCollectionNodePtr ptLink;
	HIDCollection *ptCollection;
	HIDP_UsageItem *ptFirstUsageItem;
	int iMaxNodes;
	int collectionCount;
	int firstUsageItem;
	int i;
/*
 *	Disallow Null Pointers
*/
	if ((ptLinkCollectionNodes == NULL)
	 || (piLinkCollectionNodesLength == NULL)
	 || (ptPreparsedData == NULL))
		return kHIDNullPointerErr;
	if (ptPreparsedData->hidTypeIfValid != kHIDOSType)
		return kHIDInvalidPreparsedDataErr;
/*
 *	Remember the size of the output array
*/
	iMaxNodes = *piLinkCollectionNodesLength;
	collectionCount = ptPreparsedData->collectionCount;
	*piLinkCollectionNodesLength = collectionCount;
/*
 *	Report if there's not enough room
*/
	if (collectionCount > iMaxNodes)
		return kHIDBufferTooSmallErr;
/*
 *	Copy the nodes
*/
	for (i=0; i<collectionCount; i++)
	{
		ptCollection = &ptPreparsedData->collections[i];
		ptLink = &ptLinkCollectionNodes[i];
		firstUsageItem = ptCollection->firstUsageItem;
		ptFirstUsageItem = &ptPreparsedData->usageItems[firstUsageItem];
		ptLink->collectionUsage = ptFirstUsageItem->usage;
		ptLink->collectionUsagePage = ptCollection->usagePage;
		ptLink->parent = ptCollection->parent;
		ptLink->numberOfChildren = ptCollection->children;
		ptLink->nextSibling = ptCollection->nextSibling;
		ptLink->firstChild = ptCollection->firstChild;
	}
/*
 *	Report if there wasn't enough space
*/
	if (iMaxNodes < (int)ptPreparsedData->collectionCount)
		return kHIDBufferTooSmallErr;
	return kHIDSuccess;
}


/*
 *------------------------------------------------------------------------------
 *
 * HIDGetCollectionExtendedNodes - Get the Collections Database
 *			Added by Rob Yepez to get at the data portoin of the reportItem
 *			This call is private.
 *
 *	 Input:
 *			  ptLinkCollectionNodes		  - Node Array provided by caller
 *			  piLinkCollectionNodesLength - Maximum Nodes
 *	 Output:
 *			  piLinkCollectionNodesLength - Actual number of Nodes
 *	 Returns:
 *			  kHIDSuccess		  - Success
 *			  kHIDNullPointerErr	 - Argument, Pointer was Null
 *			  HidP_NotEnoughRoom   - More Nodes than space for them
 *
 *------------------------------------------------------------------------------
*/
OSStatus HIDGetCollectionExtendedNodes( HIDCollectionExtendedNodePtr ptLinkCollectionNodes,
                                        UInt32 *piLinkCollectionNodesLength,
                                        HIDPreparsedDataRef preparsedDataRef)
{
	HIDPreparsedDataPtr ptPreparsedData = (HIDPreparsedDataPtr) preparsedDataRef;
	HIDCollectionExtendedNodePtr ptLink;
	HIDCollection *ptCollection;
	HIDP_UsageItem *ptFirstUsageItem;
	int iMaxNodes;
	int collectionCount;
	int firstUsageItem;
	int i;
/*
 *	Disallow Null Pointers
*/
	if ((ptLinkCollectionNodes == NULL)
	 || (piLinkCollectionNodesLength == NULL)
	 || (ptPreparsedData == NULL))
		return kHIDNullPointerErr;
	if (ptPreparsedData->hidTypeIfValid != kHIDOSType)
		return kHIDInvalidPreparsedDataErr;
/*
 *	Remember the size of the output array
*/
	iMaxNodes = *piLinkCollectionNodesLength;
	collectionCount = ptPreparsedData->collectionCount;
	*piLinkCollectionNodesLength = collectionCount;
/*
 *	Report if there's not enough room
*/
	if (collectionCount > iMaxNodes)
		return kHIDBufferTooSmallErr;
/*
 *	Copy the nodes
*/
	for (i=0; i<collectionCount; i++)
	{
		ptCollection = &ptPreparsedData->collections[i];
		ptLink = &ptLinkCollectionNodes[i];
		firstUsageItem = ptCollection->firstUsageItem;
		ptFirstUsageItem = &ptPreparsedData->usageItems[firstUsageItem];
		ptLink->collectionUsage = ptFirstUsageItem->usage;
		ptLink->collectionUsagePage = ptCollection->usagePage;
		ptLink->parent = ptCollection->parent;
		ptLink->numberOfChildren = ptCollection->children;
		ptLink->nextSibling = ptCollection->nextSibling;
		ptLink->firstChild = ptCollection->firstChild;
                ptLink->data = ptCollection->data;
	}
/*
 *	Report if there wasn't enough space
*/
	if (iMaxNodes < (int)ptPreparsedData->collectionCount)
		return kHIDBufferTooSmallErr;
	return kHIDSuccess;
}

