/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 22, 2024.
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
 * Date: Monday, May 13, 2024.
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
 * HIDSetUsageValueArray - Set the values for a usage
 *
 *	 Input:
 *			  reportType		   - HIDP_Input, HIDP_Output, HIDP_Feature
 *			  usagePage			   - Page Criteria
 *			  iCollection			- Collection Criteria or zero
 *			  usage				   - usage Criteria
 *			  psBuffer				- Pointer to usage Buffer
 *			  iByteLength			- Length of usage Buffer
 *			  ptPreparsedData		- Pre-Parsed Data
 *			  psReport				- An HID Report
 *			  iReportLength			- The length of the Report
 *	 Output:
 *			  piValue				- Pointer to usage Value
 *	 Returns:
 *
 *------------------------------------------------------------------------------
*/
OSStatus HIDSetUsageValueArray(HIDReportType reportType,
									HIDUsage usagePage,
									UInt32 iCollection,
									HIDUsage usage,
									UInt8 *psUsageBuffer,
									IOByteCount iByteLength,
									HIDPreparsedDataRef preparsedDataRef,
									void *psReport,
									IOByteCount iReportLength)
{
	HIDPreparsedDataPtr ptPreparsedData = (HIDPreparsedDataPtr) preparsedDataRef;
	HIDCollection *ptCollection;
	HIDReportItem *ptReportItem;
	OSStatus iStatus;
	int i;
	int iR;
	SInt32 iValue;
	int iStart;
	int iReportItem;
	UInt32 iUsageIndex;
	UInt32 iCount;
	int byteCount;
	Boolean bIncompatibleReport = false;
/*
 *	Disallow Null Pointers
*/
	if ((ptPreparsedData == NULL)
	 || (psUsageBuffer == NULL)
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
		 && HIDHasUsage(preparsedDataRef,ptReportItem,usagePage,usage,&iUsageIndex,&iCount))
		{
/*
 *			This may be the proper place
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
 *				Disallow single count variables
 *				Count is set by HasUsage
*/
				if (iCount <= 1)
					return kHIDNotValueArrayErr;
/*
 *				Write out the data
*/
				iStart = (int)(ptReportItem->startBit + (ptReportItem->globals.reportSize * iUsageIndex));
				byteCount = (int)(ptReportItem->globals.reportSize * iCount + 7)/8;
				if (byteCount > (int)iByteLength)
					byteCount = (int)iByteLength;
				for (i=0; i<byteCount; i++)
				{
					iValue = *psUsageBuffer++;
					HIDPreProcessRIValue (ptReportItem, &iValue); // error ignored
					iStatus = HIDPutData(psReport, iReportLength, iStart, 8, iValue);
					if (iStatus != kHIDSuccess)
						return iStatus;
					iStart += 8;
				}
				return kHIDSuccess;
			}
		}
	}
	if (bIncompatibleReport)
		return kHIDIncompatibleReportErr;
	return kHIDUsageNotFoundErr;
}

