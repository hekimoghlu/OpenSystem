/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 18, 2023.
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
 * Date: Monday, June 23, 2025.
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
 * HIDGetUsageValue - Get the value for a usage
 *
 *	 Input:
 *			  reportType		   - HIDP_Input, HIDP_Output, HIDP_Feature
 *			  usagePage			   - Page Criteria or zero
 *			  iCollection			- Collection Criteria or zero
 *			  usage				   - The usage to get the value for
 *			  piUsageValue			- User-supplied place to put value
 *			  ptPreparsedData		- Pre-Parsed Data
 *			  psReport				- An HID Report
 *			  iReportLength			- The length of the Report
 *	 Output:
 *			  piValue				- Pointer to usage Value
 *	 Returns:
 *
 *------------------------------------------------------------------------------
*/
OSStatus HIDGetUsageValue
		  (HIDReportType			reportType,
		   HIDUsage					usagePage,
		   UInt32					iCollection,
		   HIDUsage					usage,
		   SInt32 *					piUsageValue,
		   HIDPreparsedDataRef		preparsedDataRef,
		   void *					psReport,
		   IOByteCount              iReportLength)
{
	HIDPreparsedDataPtr ptPreparsedData = (HIDPreparsedDataPtr) preparsedDataRef;
	HIDCollection *ptCollection;
	HIDReportItem *ptReportItem;
	OSStatus iStatus;
	int iR;
	SInt32 iValue;
	int iStart;
	int iReportItem;
	UInt32 iUsageIndex;
	Boolean bIncompatibleReport = false;
/*
 *	Disallow Null Pointers
*/
	if ((ptPreparsedData == NULL)
	 || (piUsageValue == NULL)
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
 *	Search only the scope of the Collection specified
 *	Go through the ReportItems
 *	Filter on ReportType and usagePage
*/
	ptCollection = &ptPreparsedData->collections[iCollection];
	for (iR=0; iR<ptCollection->reportItemCount; iR++)
	{
		iReportItem = ptCollection->firstReportItem + iR;
		ptReportItem = &ptPreparsedData->reportItems[iReportItem];
		if (HIDIsVariable(ptReportItem, preparsedDataRef)
		 && HIDHasUsage(preparsedDataRef,ptReportItem,usagePage,usage,&iUsageIndex,NULL))
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
/*
 *				Pick up the data
*/
				iStart = (int)(ptReportItem->startBit
					   + (ptReportItem->globals.reportSize * iUsageIndex));
				iStatus = HIDGetData(psReport, iReportLength, iStart,
									   (UInt32)ptReportItem->globals.reportSize, &iValue,
									   ((ptReportItem->globals.logicalMinimum < 0)
									  ||(ptReportItem->globals.logicalMaximum < 0)));
				if (!iStatus)
					iStatus = HIDPostProcessRIValue (ptReportItem, &iValue);
				*piUsageValue = iValue;
				return iStatus;
			}
		}
	}
	if (bIncompatibleReport)
		return kHIDIncompatibleReportErr;
	return kHIDUsageNotFoundErr;
}

/*
 *------------------------------------------------------------------------------
 *
 * HIDGetScaledUsageValue - Get the value for a usage
 *
 *	 Input:
 *			  reportType		   - HIDP_Input, HIDP_Output, HIDP_Feature
 *			  usagePage			   - Page Criteria or zero
 *			  iCollection			- Collection Criteria or zero
 *			  usage				   - usage Criteria or zero
 *			  piValue				- Pointer to usage Value
 *			  ptPreparsedData		- Pre-Parsed Data
 *			  psReport				- An HID Report
 *			  iReportLength			- The length of the Report
 *	 Output:
 *			  piValue				- Pointer to usage Value
 *	 Returns:
 *
 *------------------------------------------------------------------------------
*/
OSStatus HIDGetScaledUsageValue(HIDReportType reportType,
									 HIDUsage usagePage,
									 UInt32 iCollection,
									 HIDUsage usage,
									 SInt32 *piUsageValue,
									 HIDPreparsedDataRef preparsedDataRef,
									 void *psReport,
									 IOByteCount iReportLength)
{
	HIDPreparsedDataPtr ptPreparsedData = (HIDPreparsedDataPtr) preparsedDataRef;
	HIDCollection *ptCollection;
	HIDReportItem *ptReportItem;
	OSStatus iStatus;
	int iR;
	SInt32 iValue;
	int iStart;
	int iReportItem;
	UInt32 iUsageIndex;
	Boolean bIncompatibleReport = false;
/*
 *	Disallow Null Pointers
*/
	if ((ptPreparsedData == NULL)
	 || (piUsageValue == NULL)
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
 *	Search only the scope of the Collection specified
 *	Go through the ReportItems
 *	Filter on ReportType and usagePage
*/
	ptCollection = &ptPreparsedData->collections[iCollection];
	for (iR=0; iR<ptCollection->reportItemCount; iR++)
	{
		iReportItem = ptCollection->firstReportItem + iR;
		ptReportItem = &ptPreparsedData->reportItems[iReportItem];
		if (HIDIsVariable(ptReportItem, preparsedDataRef)
		 && HIDHasUsage(preparsedDataRef,ptReportItem,usagePage,usage,&iUsageIndex,NULL))
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
/*
 *				Pick up the data
*/
				iStart = (int)(ptReportItem->startBit
					   + (ptReportItem->globals.reportSize * iUsageIndex));
				iStatus = HIDGetData(psReport, iReportLength, iStart,
									   (UInt32)ptReportItem->globals.reportSize, &iValue,
									   ((ptReportItem->globals.logicalMinimum < 0)
									  ||(ptReportItem->globals.logicalMaximum < 0)));
				if (!iStatus)
					iStatus = HIDPostProcessRIValue (ptReportItem, &iValue);
				if (iStatus != kHIDSuccess)
					return iStatus;
/*
 *				Try to scale the data
*/
				 iStatus = HIDScaleUsageValueIn(ptReportItem,iValue,&iValue);
				*piUsageValue = iValue;
				return iStatus;
			}
		}
	}
	if (bIncompatibleReport)
		return kHIDIncompatibleReportErr;
	return kHIDUsageNotFoundErr;
}

