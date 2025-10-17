/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 25, 2023.
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
 * Date: Sunday, June 15, 2025.
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
 * HIDProcessGlobalItem - Process a GlobalItem
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
OSStatus HIDProcessGlobalItem(HIDReportDescriptor *ptDescriptor, HIDPreparsedDataPtr ptPreparsedData)
{
	HIDReportSizes *ptReport;
	HIDGlobalItems *ptGlobals;
	HIDItem *ptItem;
	int reportIndex;
/*
 *	Disallow NULL Pointers
*/
	if ((ptDescriptor == NULL) || (ptPreparsedData == NULL))
		return kHIDNullPointerErr;
/*
 *	Process by tag
*/
	ptItem = &ptDescriptor->item;
	ptGlobals = &ptDescriptor->globals;
	switch (ptItem->tag)
	{
/*
 *		usage Page
*/
		case kHIDTagUsagePage:
#if 0
			// some device actually have a usage page of zero specified.  we must allow it!
			if (ptItem->unsignedValue == 0)
				return kHIDUsagePageZeroErr;
#endif
			ptGlobals->usagePage = ptItem->unsignedValue;
			break;
/*
 *		Logical Minimum
*/
		case kHIDTagLogicalMinimum:
			ptGlobals->logicalMinimum = ptItem->signedValue;
			break;
/*
 *		Logical Maximum
*/
		case kHIDTagLogicalMaximum:
			ptGlobals->logicalMaximum = ptItem->signedValue;
			break;
/*
 *		Physical Minimum
*/
		case kHIDTagPhysicalMinimum:
			ptGlobals->physicalMinimum = ptItem->signedValue;
			break;
/*
 *		Physical Maximum
*/
		case kHIDTagPhysicalMaximum:
			ptGlobals->physicalMaximum = ptItem->signedValue;
			break;
/*
 *		Unit Exponent
*/
		case kHIDTagUnitExponent:
			ptGlobals->unitExponent = ptItem->signedValue;
			break;
/*
 *		Unit
*/
		case kHIDTagUnit:
			ptGlobals->units = ptItem->unsignedValue;
			break;
/*
 *		Report Size in Bits
*/
		case kHIDTagReportSize:
			ptGlobals->reportSize = ptItem->unsignedValue;
#if 0
			// some device actually have a report size of zero specified.  we must allow it!
			if (ptGlobals->reportSize == 0)
				return kHIDReportSizeZeroErr;
#endif
			break;
/*
 *		Report ID
*/
		case kHIDTagReportID:
#if 0
			// some device actually have a report id of zero specified.  we must allow it!
			if (ptItem->unsignedValue == 0)
				return kHIDReportIDZeroErr;
#endif
/*
 *			Look for the Report ID in the table
*/
			reportIndex = 0;
			while ((reportIndex < (int)ptPreparsedData->reportCount)
				&& (ptPreparsedData->reports[reportIndex].reportID != (SInt32)ptItem->unsignedValue))
				reportIndex++;
/*
 *			Initialize the entry if it's new and there's room for it
 *			  Start with 8 bits for the Report ID
*/
			if (reportIndex == (int)ptPreparsedData->reportCount)
			{
				ptReport = &ptPreparsedData->reports[ptPreparsedData->reportCount++];
				ptReport->reportID = ptItem->unsignedValue;
				ptReport->inputBitCount = 8;
				ptReport->outputBitCount = 8;
				ptReport->featureBitCount = 8;
			}
/*
 *			Remember which report is being processed
*/
			ptGlobals->reportID = ptItem->unsignedValue;
			ptGlobals->reportIndex = reportIndex;
			break;
/*
 *		Report Count
*/
		case kHIDTagReportCount:
#if 0
			// some device actually have a report count of zero specified.  we must allow it!
			if (ptItem->unsignedValue == 0)
				return kHIDReportCountZeroErr;
#endif				
			ptGlobals->reportCount = ptItem->unsignedValue;
			break;
/*
 *		Push Globals
*/
		case kHIDTagPush:
			ptDescriptor->globalsStack[ptDescriptor->globalsNesting++] = ptDescriptor->globals;
			break;
/*
 *		Pop Globals
*/
		case kHIDTagPop:
			ptDescriptor->globals = ptDescriptor->globalsStack[--ptDescriptor->globalsNesting];
			break;
	}
	return kHIDSuccess;
}

