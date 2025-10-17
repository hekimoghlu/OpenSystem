/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 2, 2025.
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
 * Date: Friday, July 22, 2022.
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
 * HIDGetButtons - Get the state of the buttons for a Page
 *
 *	 Input:
 *			  reportType		   - HIDP_Input, HIDP_Output, HIDP_Feature
 *			  usagePage			   - Page Criteria or zero
 *			  iCollection			- Collection Criteria or zero
 *			  piUsageList			- Usages for pressed buttons
 *			  piUsageListLength		- Max entries in UsageList
 *			  ptPreparsedData		- Pre-Parsed Data
 *			  psReport				- An HID Report
 *			  iReportLength			- The length of the Report
 *	 Output:
 *			  piValue				- Pointer to usage Value
 *	 Returns:
 *
 *------------------------------------------------------------------------------
*/
OSStatus 
HIDGetButtons  (HIDReportType			reportType,
				UInt32					iCollection,
				HIDUsageAndPagePtr		ptUsageList,
				UInt32 *				piUsageListLength,
				HIDPreparsedDataRef 	preparsedDataRef,
				void *					psReport,
				IOByteCount             iReportLength)
{
	HIDPreparsedDataPtr ptPreparsedData = (HIDPreparsedDataPtr) preparsedDataRef;
	HIDCollection *ptCollection;
	HIDReportItem *ptReportItem;
	int iR, iE;
	SInt32 iValue;
	int iStart;
	int iReportItem;
	int iMaxUsages;
	HIDUsageAndPage tUsageAndPage;
	
/*
 *	Disallow Null Pointers
*/
	if ((ptPreparsedData == NULL)
	 || (ptUsageList == NULL)
	 || (piUsageListLength == NULL)
	 || (psReport == NULL))
		return kHIDNullPointerErr;
	if (ptPreparsedData->hidTypeIfValid != kHIDOSType)
		return kHIDInvalidPreparsedDataErr;
/*
 *	Save the UsageList size
*/
	iMaxUsages = *piUsageListLength;
	*piUsageListLength = 0;
/*
 *	Search only the scope of the Collection specified
 *	Go through the ReportItems
 *	Filter on ReportType
*/
	ptCollection = &ptPreparsedData->collections[iCollection];
	for (iR=0; iR<ptCollection->reportItemCount; iR++)
	{
		iReportItem = ptCollection->firstReportItem + iR;
		ptReportItem = &ptPreparsedData->reportItems[iReportItem];
		if ((ptReportItem->reportType == reportType)
		 && HIDIsButton(ptReportItem, preparsedDataRef))
		{
/*
 *			Save Arrays and Bitmaps
*/
			iStart = ptReportItem->startBit;
			for (iE=0; iE<ptReportItem->globals.reportCount; iE++)
			{
				OSStatus status = 0;
				iValue = 0;
				
				if ((ptReportItem->dataModes & kHIDDataArrayBit) == kHIDDataArray)
				{
					status = HIDGetData(psReport, iReportLength, iStart, (UInt32)ptReportItem->globals.reportSize, &iValue, false);
					if (!status)
						status = HIDPostProcessRIValue (ptReportItem, &iValue);
					if (status) return status;
					
					iStart += ptReportItem->globals.reportSize;
					HIDUsageAndPageFromIndex(preparsedDataRef,ptReportItem,ptReportItem->globals.logicalMinimum+iE,&tUsageAndPage);
					if (*piUsageListLength >= (UInt32)iMaxUsages)
						return kHIDBufferTooSmallErr;
					ptUsageList[(*piUsageListLength)++] = tUsageAndPage;
				}
				else
				{
					status = HIDGetData(psReport, iReportLength, iStart, 1, &iValue, false);
					if (!status)
						status = HIDPostProcessRIValue (ptReportItem, &iValue);
					if (status) return status;

					iStart++;
					if (iValue != 0)
					{
						HIDUsageAndPageFromIndex(preparsedDataRef,ptReportItem,ptReportItem->globals.logicalMinimum+iE,&tUsageAndPage);
						if (*piUsageListLength >= (UInt32)iMaxUsages)
							return kHIDBufferTooSmallErr;
						ptUsageList[(*piUsageListLength)++] = tUsageAndPage;
					}
				}
			}
		}
	}
	return kHIDSuccess;
}

