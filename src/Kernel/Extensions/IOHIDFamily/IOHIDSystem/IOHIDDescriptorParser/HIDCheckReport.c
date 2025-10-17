/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 11, 2025.
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
 * Date: Friday, June 28, 2024.
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
 * HIDCheckReport - Check the Report ID, Type, and Length
 *
 *	 Input:
 *			  reportType		   - The Specified Report Type
 *			  ptPreparsedData		- The Preparsed Data
 *			  ptReportItem			- The Report Item
 *			  psReport				- The Report
 *			  iReportLength			- The Report Length
 *	 Output:
 *	 Returns:
 *			  kHIDSuccess, HidP_IncompatibleReportID,
 *			  kHIDInvalidReportLengthErr, kHIDInvalidReportTypeErr
 *
 *------------------------------------------------------------------------------
*/
OSStatus HIDCheckReport(HIDReportType reportType, HIDPreparsedDataRef preparsedDataRef,
							 HIDReportItem *ptReportItem, void *report, IOByteCount iReportLength)
{
	HIDPreparsedDataPtr ptPreparsedData = (HIDPreparsedDataPtr) preparsedDataRef;
	int reportID, reportIndex;
	int iExpectedLength;
	UInt8 * psReport = (UInt8 *)report;
/*
 *	See if this is the correct Report ID
*/
	reportID = psReport[0]&0xFF;
	if ((ptPreparsedData->reportCount > 1)
	 && (reportID != ptReportItem->globals.reportID))
		return kHIDIncompatibleReportErr;
/*
 *	See if this is the correct ReportType
*/
	if (reportType != ptReportItem->reportType)
		return kHIDIncompatibleReportErr;
/*
 *	Check for the correct Length for the Type
*/
	reportIndex = ptReportItem->globals.reportIndex;
	switch(reportType)
	{
		case kHIDInputReport:
			iExpectedLength = (ptPreparsedData->reports[reportIndex].inputBitCount + 7)/8;
			break;
		case kHIDOutputReport:
			iExpectedLength = (ptPreparsedData->reports[reportIndex].outputBitCount + 7)/8;
			break;
		case kHIDFeatureReport:
			iExpectedLength = (ptPreparsedData->reports[reportIndex].featureBitCount + 7)/8;
			break;
		default:
			return kHIDInvalidReportTypeErr;
	}
	if (iExpectedLength > (int)iReportLength)
		return kHIDInvalidReportLengthErr;
	return kHIDSuccess;
}

