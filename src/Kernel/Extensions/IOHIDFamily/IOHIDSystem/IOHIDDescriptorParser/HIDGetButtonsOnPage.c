/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 10, 2022.
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
 * Date: Thursday, February 15, 2024.
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
 * HIDGetButtonsOnPage - Get the state of the buttons for a Page
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
OSStatus HIDGetButtonsOnPage(HIDReportType reportType,
						   HIDUsage usagePage,
						   UInt32 iCollection,
						   HIDUsage *piUsageList,
						   UInt32 *piUsageListLength,
						   HIDPreparsedDataRef preparsedDataRef,
						   void *psReport,
						   IOByteCount iReportLength)
{
	HIDPreparsedDataPtr ptPreparsedData = (HIDPreparsedDataPtr) preparsedDataRef;
	HIDUsageAndPage tUsageAndPage;
	HIDCollection *ptCollection;
	HIDReportItem *ptReportItem;
	OSStatus iStatus;
	int iR, iE;
	SInt32 iValue;
	int iStart;
	int iMaxUsages;
	int iReportItem;
	Boolean bIncompatibleReport = false;
	Boolean butNotReally = false;
/*
 *	Disallow Null Pointers
*/
	if ((ptPreparsedData == NULL)
	 || (piUsageList == NULL)
	 || (piUsageListLength == NULL)
	 || (psReport == NULL))
		return kHIDNullPointerErr;
	if (ptPreparsedData->hidTypeIfValid != kHIDOSType)
		return kHIDInvalidPreparsedDataErr;
/*
 *	The Collection must be in range
*/
	if (iCollection >= ptPreparsedData->collectionCount)
		return kHIDBadParameterErr;
/*
 *	Save the size of the list
*/
	iMaxUsages = *piUsageListLength;
	*piUsageListLength = 0;
/*
 *	Search only the scope of the Collection specified
 *	Go through the ReportItems
 *	Filter on ReportType and usagePage
*/
	ptCollection = &ptPreparsedData->collections[iCollection];
	for (iR=0; iR<ptCollection->reportItemCount; iR++)
	{
		iReportItem = ptCollection->firstReportItem + iR;
		ptReportItem = &ptPreparsedData->reportItems[iReportItem];
		if (HIDIsButton(ptReportItem, preparsedDataRef))
		{
/*
 *			This may be the proper data to get
 *			Let's check for the proper Report ID, Type, and Length
*/
			iStatus = HIDCheckReport(reportType,preparsedDataRef,ptReportItem,
									   psReport,iReportLength);
/*
 *			The Report ID or Type may not match.
 *			This may not be an error (yet)
*/
			if (iStatus == kHIDIncompatibleReportErr)
				bIncompatibleReport = true;
			else if (iStatus != kHIDSuccess)
				return iStatus;
			else
			{
				butNotReally = true;
/*
 *				Save Array Buttons
*/
				iStart = ptReportItem->startBit;
				for (iE=0; iE<ptReportItem->globals.reportCount; iE++)
				{
					if ((ptReportItem->dataModes & kHIDDataArrayBit) == kHIDDataArray)
					{
						iStatus = HIDGetData(psReport, iReportLength, iStart,
									 (UInt32)ptReportItem->globals.reportSize,
									 &iValue, false);
						if (!iStatus)
							HIDPostProcessRIValue (ptReportItem, &iValue); // error ignored
						HIDUsageAndPageFromIndex(preparsedDataRef,
									 ptReportItem,
									 iValue-ptReportItem->globals.logicalMinimum,
									 &tUsageAndPage);
						iStart += ptReportItem->globals.reportSize;
						if (usagePage == tUsageAndPage.usagePage)
						{
							if (*piUsageListLength >= (UInt32)iMaxUsages)
								return kHIDBufferTooSmallErr;
							piUsageList[(*piUsageListLength)++] = iValue;
						}
					}
/*
 *					Save Bitmapped Buttons
*/
					else
					{
						iStatus = HIDGetData(psReport, iReportLength, iStart, 1, &iValue, false);
						if (!iStatus)
							iStatus = HIDPostProcessRIValue (ptReportItem, &iValue);
						iStart++;
						if (!iStatus && iValue != 0)
						{
							HIDUsageAndPageFromIndex(preparsedDataRef,ptReportItem,iE,&tUsageAndPage);
							if (usagePage == tUsageAndPage.usagePage)
							{
								if (*piUsageListLength >= (UInt32)iMaxUsages)
									return kHIDBufferTooSmallErr;
								piUsageList[(*piUsageListLength)++] = tUsageAndPage.usage;
							}
						}
					}
				}
			}
		}
	}
/*
 *	If nothing was returned then change the status
*/
	if (*piUsageListLength == 0)
	{
		// If any of the report items were not the right type, we have set the bIncompatibleReport flag.
		// However, if any of the report items really were the correct type, we have done our job of checking
		// and really didn't find a usage. Don't let the bIncompatibleReport flag wipe out our valid test.
		if (bIncompatibleReport && !butNotReally)
			return kHIDIncompatibleReportErr;
		return kHIDUsageNotFoundErr;
	}
	return kHIDSuccess;
}

