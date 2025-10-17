/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 14, 2022.
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
 * Date: Sunday, January 1, 2023.
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
 * HidP_UsageFromIndex
 *
 *	 Input:
 *			  ptPreparsedData		- The Preparsed Data
 *			  ptReportItem			- The Report Item
 *			  usagePage			   - The usage Page to find
 *			  usage				   - The usage to find
 *			  piIndex(optional)		- The usage Index pointer (Can be used to tell
 *										which bits in an array correspond to that usage.)
 *			  piCount(optional)		- The usage Count pointer (Can be used to tell 
 *										how many items will be in a report.)
 *	 Output:
 *			  piIndex				- The usage Index
 *	 Returns:
 *			  The usage
 *
 *------------------------------------------------------------------------------
*/
Boolean HIDHasUsage (HIDPreparsedDataRef preparsedDataRef,
					   HIDReportItem *ptReportItem,
					   HIDUsage usagePage, HIDUsage usage,
					   UInt32 *piIndex, UInt32 *piCount)
{
	HIDPreparsedDataPtr ptPreparsedData = (HIDPreparsedDataPtr) preparsedDataRef;
	int iUsageItem;
	UInt32 iUsageIndex;
	int iUsages;
	int i;
	SInt32 iCountsLeft;
	HIDP_UsageItem *ptUsageItem;
	Boolean bOnPage;
/*
 *	Disallow Null Pointers
*/
	if ((ptPreparsedData == NULL)
	 || (ptReportItem == NULL))
		return 0;
	if (ptPreparsedData->hidTypeIfValid != kHIDOSType)
		return 0;
/*
 *	Look through the usage Items for this usage
*/
	iUsageItem = ptReportItem->firstUsageItem;
	iUsageIndex = 0;
	for (i=0; i<ptReportItem->usageItemCount; i++)
	{
/*
 *	   Each usage Item is either a usage or a usage range
*/
		ptUsageItem = &ptPreparsedData->usageItems[iUsageItem++];
		bOnPage = ((usagePage == 0) || (usagePage == ptUsageItem->usagePage));
		if (ptUsageItem->isRange)
		{
/*
 *			For usage Ranges
 *			  If the index is in the range
 *				then return the usage
 *			  Otherwise adjust the index by the size of the range
*/
			if ((usage >= (HIDUsage)ptUsageItem->usageMinimum)
			 && (usage <= (HIDUsage)ptUsageItem->usageMaximum))
			{
				if (piIndex != NULL)
					*piIndex = iUsageIndex + (ptUsageItem->usageMinimum - usage);
/*
 *				If this usage is the last one for this ReportItem
 *				  then it gets all of the remaining reportCount
*/
				if (piCount != NULL)
				{
					// piCount is going to be used to find which element in a button array is
					// the one that returns the value for that usage.
					if (((i+1) == ptReportItem->usageItemCount)
					 && (usage == (HIDUsage)ptUsageItem->usageMaximum))
					{
						// Hmm, the same logic in the non-range case below was wrong. But things
						// seem to be working for finding buttons, so i am not changing it here.
						// However, we have made some changes to range calculations that may no
						// longer require that -1 here either. Heads up!
						iCountsLeft = ptReportItem->globals.reportCount - iUsageIndex - 1;
						if (iCountsLeft > 1)
							*piCount = iCountsLeft;
						else
							*piCount = 1;
					}
					else
						*piCount = 1;
				}
				if (bOnPage)
					return true;
			}
			iUsages = ptUsageItem->usageMaximum - ptUsageItem->usageMinimum;
			if (iUsages < 0)
				iUsages = -iUsages;
			iUsages++;		// Add off by one adjustment AFTER sign correction.
			iUsageIndex += iUsages;
		}
		else
		{
/*
 *			For Usages
 *			If the index is zero
 *			  then return this usage
 *			Otherwise one less to index through
*/
			if (usage == ptUsageItem->usage)
			{
				if (piIndex != NULL)
					*piIndex = iUsageIndex;
				if (piCount != NULL)
				{
					if ((i+1) == ptReportItem->usageItemCount)
					{
						// Keithen does not understand the logic of iCountsLeft.
						// In Radar #2579612 we come through here for HIDGetUsageValueArray
						// and HIDGetSpecificValueCaps. In both cases piCount that is returned
						// should be the reportCount without the -1.
//						iCountsLeft = ptReportItem->globals.reportCount - iUsageIndex - 1;
						iCountsLeft = ptReportItem->globals.reportCount - iUsageIndex;
						if (iCountsLeft > 1)
							*piCount = iCountsLeft;
						else
						   *piCount = 1;
					}
					else
						*piCount = 1;
				}
				if (bOnPage)
					return true;
			}
			iUsageIndex++;
		}
	}
	return false;
}

