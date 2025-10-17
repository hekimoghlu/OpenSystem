/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 9, 2023.
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
 * Date: Thursday, April 17, 2025.
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
 * HIDGetNextButtonInfo - Get report id and collection for a button. In keeping
 *								with USBGetNextInterface, we find the usage in the
 *								next collection, so that you can find usages that
 *								have the same usage and usage page.
 *
 *	 Input:
 *			  reportType			- HIDP_Input, HIDP_Output, HIDP_Feature
 *			  usagePage				- Page Criteria or zero
 *			  usage					- The usage to get the information for
 *			  collection			- Starting Collection Criteria or zero
 *			  preparsedDataRef		- Pre-Parsed Data
 *	 Output:
 *			  collection			- Final Collection Criteria or no change
 *			  reportID				- Report ID or no change
 *	 Returns:
 *
 *------------------------------------------------------------------------------
*/
OSStatus HIDGetNextButtonInfo
					   (HIDReportType			reportType,
						HIDUsage				usagePage,
						HIDUsage				usage,
						UInt32 *				collection,
						UInt8 *					reportID,
						HIDPreparsedDataRef		preparsedDataRef)
{
	HIDPreparsedDataPtr ptPreparsedData = (HIDPreparsedDataPtr)preparsedDataRef;
	HIDReportItem *ptReportItem;
	UInt32 iCollection;
	UInt32 newCollection = 0xFFFFFFFF;
	int iR;
	UInt8 newReportID = 0;
	OSStatus iStatus = kHIDUsageNotFoundErr;

	//Disallow Null Pointers

	if ((ptPreparsedData == NULL) || (collection == NULL) || (reportID == NULL))
		return kHIDNullPointerErr;
	if (ptPreparsedData->hidTypeIfValid != kHIDOSType)
		return kHIDInvalidPreparsedDataErr;

	// The Collection must be in range

	iCollection = *collection;
	if (iCollection >= ptPreparsedData->collectionCount)
		return kHIDBadParameterErr;

	// HIDGetNextButtonInfo is different from HIDGetButton in how it treats
	// the collection parameter. HIDGetButton will only look at report items that
	// are within the collection and can therefore limit it's searches to starting at
	// ptPreparsedData->collections[iCollection]->firstReportItem and only check
	// ptPreparsedData->collections[iCollection]->reportItemCount. Since we want to 
	// find the NEXT collection as well, we need to cycle through all of the reports.

	for (iR = 0; iR < (int)ptPreparsedData->reportItemCount; iR++)
	{
		SInt32 minUsage;
		SInt32 maxUsage;
		HIDP_UsageItem thisUsage;

		ptReportItem = &ptPreparsedData->reportItems[iR];
		
		thisUsage = ptPreparsedData->usageItems[ptReportItem->firstUsageItem];

		if (thisUsage.isRange)
		{
			minUsage = thisUsage.usageMinimum;
			maxUsage = thisUsage.usageMaximum;
		}
		else
		{
			minUsage = thisUsage.usage;
			maxUsage = thisUsage.usage;
		}

		if (ptReportItem->reportType == reportType &&
			(usagePage == 0 || ptReportItem->globals.usagePage == usagePage) &&
			(usage >= (HIDUsage)minUsage && usage <= (HIDUsage)maxUsage) &&
			ptReportItem->parent > (SInt32)iCollection &&
			HIDIsButton(ptReportItem, preparsedDataRef))
		{
			if (ptReportItem->parent < (SInt32)newCollection)
			{
				newCollection = ptReportItem->parent;
				newReportID = iR;
				iStatus = 0;
			}
		}
	}
		
	if (!iStatus)
	{
		*reportID = newReportID;
		*collection = newCollection;
	}

	return iStatus;
}

