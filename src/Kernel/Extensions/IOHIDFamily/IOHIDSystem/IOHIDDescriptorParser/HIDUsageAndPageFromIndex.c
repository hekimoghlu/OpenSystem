/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 10, 2025.
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
 * Date: Monday, October 30, 2023.
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
 * HIDUsageAndPageFromIndex
 *
 *	 Input:
 *			  ptPreparsedData		- The Preparsed Data
 *			  ptReportItem			- The Report Item
 *			  index				   - The usage Index
 *			  ptUsageAndPage		- The usage And Page
 *	 Output:
 *	 Returns:
 *
 *------------------------------------------------------------------------------
*/
void HIDUsageAndPageFromIndex (HIDPreparsedDataRef preparsedDataRef,
								 HIDReportItem *ptReportItem, UInt32 index,
								 HIDUsageAndPage *ptUsageAndPage)
{
	HIDPreparsedDataPtr ptPreparsedData = (HIDPreparsedDataPtr) preparsedDataRef;
	HIDP_UsageItem *ptUsageItem = NULL;
	int iUsageItem;
	int iUsages;
	int i;

/*
 *	Disallow NULL Pointers
*/
	if (ptUsageAndPage == NULL)
	{
		return;	// kHIDNullPointerErr;
	}
	if ((ptReportItem == NULL) || (ptPreparsedData == NULL))
	{
        ptUsageAndPage->usagePage = 0;
		return;	// kHIDNullPointerErr;
	}
    
/*
 *	Index through the usage Items for this ReportItem
*/
	iUsageItem = ptReportItem->firstUsageItem;
	for (i=0; i<ptReportItem->usageItemCount; i++)
	{
/*
 *		Each usage Item is either a usage or a usage range
*/
		ptUsageItem = &ptPreparsedData->usageItems[iUsageItem++];
		if (ptUsageItem->isRange)
		{
/*
 *			For usage Ranges
 *			  If the index is in the range
 *				then return the usage
 *			  Otherwise adjust the index by the size of the range
*/
			iUsages = ptUsageItem->usageMaximum - ptUsageItem->usageMinimum;
			if (iUsages < 0)
				iUsages = -iUsages;
			iUsages++;		// Add off by one adjustment AFTER sign correction.
			if (iUsages > (int)index)
			{
				ptUsageAndPage->usagePage = ptUsageItem->usagePage;
				ptUsageAndPage->usage = ptUsageItem->usageMinimum + index;
				return;
			}
			index -= iUsages;
		}
		else
		{
/*
 *			For Usages
 *			If the index is zero
 *			  then return this usage
 *			Otherwise one less to index through
*/
			if (index-- == 0)
			{
				ptUsageAndPage->usagePage = ptUsageItem->usagePage;
				ptUsageAndPage->usage = ptUsageItem->usage;
				return;
			}
		}
	}
	if (ptUsageItem != NULL)
	{
		ptUsageAndPage->usagePage = ptUsageItem->usagePage;
		if (ptUsageItem->isRange)
			ptUsageAndPage->usage = ptUsageItem->usageMaximum;
		else
			ptUsageAndPage->usage = ptUsageItem->usage;
	}
}

