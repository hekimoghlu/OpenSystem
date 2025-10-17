/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 6, 2025.
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
 * Date: Tuesday, October 22, 2024.
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
 * HIDSetButton - Set the state of a button for a Page
 *
 *	 Input:
 *			  reportType		   - HIDP_Input, HIDP_Output, HIDP_Feature
 *			  usagePage			   - Page Criteria or zero
 *			  iCollection			- Collection Criteria or zero
 *			  usage				   - Usages for pressed button
 *			  ptPreparsedData		- Pre-Parsed Data
 *			  psReport				- An HID Report
 *			  iReportLength			- The length of the Report
 *	 Output:
 *	 Returns:
 *
 *------------------------------------------------------------------------------
*/
OSStatus HIDSetButton  (HIDReportType 			reportType,
						HIDUsage				usagePage,
						UInt32					collection,
						HIDUsage				usage,
						HIDPreparsedDataRef		preparsedDataRef,
						void *					report,
						IOByteCount				reportLength)
{
	HIDPreparsedDataPtr ptPreparsedData = (HIDPreparsedDataPtr) preparsedDataRef;
	HIDCollection *ptCollection;
	HIDReportItem *ptReportItem;
	OSStatus iStatus;
	int iR, iX;
	SInt32 data;
	int iStart;
	int iReportItem;
	UInt32 iUsageIndex;
	Boolean bIncompatibleReport = false;
	Boolean butNotReally = false;
/*
 *	Disallow Null Pointers
*/
	if ((ptPreparsedData == NULL)
	 || (report == NULL))
		return kHIDNullPointerErr;
	if (ptPreparsedData->hidTypeIfValid != kHIDOSType)
		return kHIDInvalidPreparsedDataErr;
/*
 *	The Collection must be in range
*/
	if (collection >= ptPreparsedData->collectionCount)
		return kHIDBadParameterErr;
/*
 *	Search only the scope of the Collection specified
 *	Go through the ReportItems
 *	Filter on ReportType and usagePage
*/
	ptCollection = &ptPreparsedData->collections[collection];
	for (iR=0; iR<ptCollection->reportItemCount; iR++)
	{
		iReportItem = ptCollection->firstReportItem + iR;
		ptReportItem = &ptPreparsedData->reportItems[iReportItem];
		if (HIDIsButton(ptReportItem, preparsedDataRef)
		 && HIDHasUsage(preparsedDataRef,ptReportItem,usagePage,usage,&iUsageIndex,NULL))
		{
/*
 *			This may be the proper data to get
 *			Let's check for the proper Report ID, Type, and Length
*/
			iStatus = HIDCheckReport(reportType,preparsedDataRef,ptReportItem,report,reportLength);
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
 *				Save Arrays
*/
				if ((ptReportItem->dataModes & kHIDDataArrayBit) == kHIDDataArray)
				{
					for (iX=0; iX<ptReportItem->globals.reportCount; iX++)
					{
						iStart = (int)(ptReportItem->startBit + (ptReportItem->globals.reportSize * iX));
						iStatus = HIDGetData(report, reportLength, iStart,
											   (UInt32)ptReportItem->globals.reportSize, &data, true);
						if (!iStatus)
							iStatus = HIDPostProcessRIValue (ptReportItem, &data);
						if (iStatus != kHIDSuccess)
							return iStatus;
						// if not already in the list, add it (is this code right??)
						if (data == 0)
							return HIDPutData(report, reportLength, iStart,
												(UInt32)ptReportItem->globals.reportSize,
												iUsageIndex + ptReportItem->globals.logicalMinimum);
					}
					return kHIDBufferTooSmallErr;
				}
/*
 *				Save Bitmaps
*/
				else if (ptReportItem->globals.reportSize == 1)
				{
					iStart = (int)(ptReportItem->startBit + (ptReportItem->globals.reportSize * iUsageIndex));
					// should we call HIDPreProcessRIValue here?
					// we are passing '-1' as trhe value, is this right? Some hack to set the right bit to 1?
					iStatus = HIDPutData(report, reportLength, iStart, (UInt32)ptReportItem->globals.reportSize, -1);
					if (iStatus != kHIDSuccess)
						return iStatus;
					return kHIDSuccess;
				}
			}
		}
	}
	// If any of the report items were not the right type, we have set the bIncompatibleReport flag.
	// However, if any of the report items really were the correct type, we have done our job of checking
	// and really didn't find a usage. Don't let the bIncompatibleReport flag wipe out our valid test.
	if (bIncompatibleReport && !butNotReally)
		return kHIDIncompatibleReportErr;
	return kHIDUsageNotFoundErr;
}

/*
 *------------------------------------------------------------------------------
 *
 * HIDSetButtons - Set the state of the buttons for a Page
 *
 *	 Input:
 *			  reportType		   - HIDP_Input, HIDP_Output, HIDP_Feature
 *			  usagePage			   - Page Criteria or zero
 *			  collection			- Collection Criteria or zero
 *			  piUsageList			- Usages for pressed buttons
 *			  piUsageListLength		- Max entries in UsageList
 *			  ptPreparsedData		- Pre-Parsed Data
 *			  report				- An HID Report
 *			  reportLength			- The length of the Report
 *	 Output:
 *			  piValue				- Pointer to usage Value
 *	 Returns:
 *
 *------------------------------------------------------------------------------
*/
OSStatus
HIDSetButtons			   (HIDReportType			reportType,
							HIDUsage				usagePage,
							UInt32					collection,
							HIDUsage *				usageList,
							UInt32 *				usageListSize,
							HIDPreparsedDataRef		preparsedDataRef,
							void *					report,
							IOByteCount				reportLength)
{
	HIDPreparsedDataPtr ptPreparsedData = (HIDPreparsedDataPtr) preparsedDataRef;
	OSStatus iStatus;
	int iUsages;
	int usage;
/*
 *	Disallow Null Pointers
*/
	if ((ptPreparsedData == NULL)
	 || (usageList == NULL)
	 || (usageListSize == NULL)
	 || (report == NULL))
		return kHIDNullPointerErr;
	if (ptPreparsedData->hidTypeIfValid != kHIDOSType)
		return kHIDInvalidPreparsedDataErr;
/*
 *	Save the usage List Length
*/
	iUsages = *usageListSize;
/*
 *	Write them out one at a time
*/
	for (usage=0; usage<iUsages; usage++)
	{
		*usageListSize = usage;
		iStatus = HIDSetButton(reportType, usagePage, collection,
								usageList[usage], preparsedDataRef,
								report, reportLength);
		if (iStatus != kHIDSuccess)
			return iStatus;
	}
	return kHIDSuccess;
}

