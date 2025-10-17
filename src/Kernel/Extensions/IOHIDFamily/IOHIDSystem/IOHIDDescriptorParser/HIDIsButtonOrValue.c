/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 14, 2025.
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
 * Date: Wednesday, April 23, 2025.
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
 *-----------------------------------------------------------------------------
 *
 * HIDIsButton - Is the data button(s)?
 *
 *	 Input:
 *			  ptReportItem			- Input/Output/Feature
 *	 Output:
 *	 Returns:
 *			  Boolean
 *
 *-----------------------------------------------------------------------------
*/
Boolean HIDIsButton(HIDReportItem *ptReportItem, HIDPreparsedDataRef preparsedDataRef)
{
	HIDPreparsedDataPtr ptPreparsedData = (HIDPreparsedDataPtr) preparsedDataRef;

/*
 *	Disallow Null Pointers
*/
	if (ptReportItem==NULL)
		return false;
/*
 *	Remove items that are constant and have no usage
 */
	if ((ptReportItem->dataModes & kHIDDataConstantBit) == kHIDDataConstant)
	{
		// if has no usages, then bit filler
		if (ptReportItem->usageItemCount == 0)
			return false;
		
		// also check to see if there is a usage, but it is zero
		
		// if the first usage item is range, then check that one
		// (we will not worry about report items with multiple zero usages, 
		//  as I dont think that is a case that makes sense)
		if (ptReportItem->firstUsageItem < (SInt32)ptPreparsedData->usageItemCount)
		{
			HIDP_UsageItem * ptUsageItem = &ptPreparsedData->usageItems[ptReportItem->firstUsageItem];
			
			// if it is a range usage, with both zero usages 
			if ((ptUsageItem->isRange && ptUsageItem->usageMinimum == 0 && ptUsageItem->usageMaximum == 0) &&
				// or not a range, and zero usage
				(!ptUsageItem->isRange && ptUsageItem->usage == 0))
				// then this is bit filler
				return false;
		}
	}

/*
 *	Arrays and 1-bit Variables
*/
	return (((ptReportItem->dataModes & kHIDDataArrayBit) == kHIDDataArray)
	   || (ptReportItem->globals.reportSize == 1));
}

/*
 *-----------------------------------------------------------------------------
 *
 * HIDIsVariable - Is the data variable(s)?
 *
 *	 Input:
 *			  ptReportItem			- Input/Output/Feature
 *	 Output:
 *	 Returns:
 *			  Boolean
 *
 *-----------------------------------------------------------------------------
*/
Boolean HIDIsVariable(HIDReportItem *ptReportItem, HIDPreparsedDataRef preparsedDataRef)
{
	HIDPreparsedDataPtr ptPreparsedData = (HIDPreparsedDataPtr) preparsedDataRef;

/*
 *	Disallow Null Pointers
*/
	if (ptReportItem==NULL)
		return false;

/*
 *	Remove items that are constant and have no usage
 */
	if ((ptReportItem->dataModes & kHIDDataConstantBit) == kHIDDataConstant)
	{
		// if has no usages, then bit filler
		if (ptReportItem->usageItemCount == 0)
			return false;
		
		// also check to see if there is a usage, but it is zero
		
		// if the first usage item is range, then check that one
		// (we will not worry about report items with multiple zero usages, 
		//  as I dont think that is a case that makes sense)
		if (ptReportItem->firstUsageItem < (SInt32)ptPreparsedData->usageItemCount)
		{
			HIDP_UsageItem * ptUsageItem = &ptPreparsedData->usageItems[ptReportItem->firstUsageItem];
			
			// if it is a range usage, with both zero usages 
			if ((ptUsageItem->isRange && ptUsageItem->usageMinimum == 0 && ptUsageItem->usageMaximum == 0) &&
				// or not a range, and zero usage
				(!ptUsageItem->isRange && ptUsageItem->usage == 0))
				// then this is bit filler
				return false;
		}
	}

/*
 *	Multi-bit Variables
*/
	return (((ptReportItem->dataModes & kHIDDataArrayBit) != kHIDDataArray)
	   && (ptReportItem->globals.reportSize != 1));
}

