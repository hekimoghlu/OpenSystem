/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 1, 2024.
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
 * Date: Monday, September 29, 2025.
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
 * HIDGetReportLength - Get the length of a report
 *
 *	 Input:
 *				reportType			- HIDP_Input, HIDP_Output, HIDP_Feature
 *				reportID			- Desired Report
 *				preparsedDataRef	- opaque Pre-Parsed Data
 *	 Output:
 *				reportLength		- The length of the Report
 *	 Returns:
 *				status				kHIDNullPointerErr, kHIDInvalidPreparsedDataErr,
 *									kHIDUsageNotFoundErr
 *
 *------------------------------------------------------------------------------
*/
OSStatus HIDGetReportLength(HIDReportType reportType,
							UInt8 reportID,
							IOByteCount * reportLength,
							HIDPreparsedDataRef preparsedDataRef)
{
	HIDPreparsedDataPtr ptPreparsedData = (HIDPreparsedDataPtr)preparsedDataRef;
	IOByteCount dataLength = 0;
	OSStatus iStatus = kHIDUsageNotFoundErr;
	int iR;

	// Disallow Null Pointers.

	if (ptPreparsedData == NULL || reportLength == NULL)
		return kHIDNullPointerErr;
	if (ptPreparsedData->hidTypeIfValid != kHIDOSType)
		return kHIDInvalidPreparsedDataErr;
		
	// Go through the Reports.

	for (iR = 0; iR < (int)ptPreparsedData->reportCount; iR++)
	{
		if (ptPreparsedData->reports[iR].reportID == reportID)
		{
			switch(reportType)
			{
				case kHIDInputReport:
					dataLength = (ptPreparsedData->reports[iR].inputBitCount + 7)/8;
					break;
				case kHIDOutputReport:
					dataLength = (ptPreparsedData->reports[iR].outputBitCount + 7)/8;
					break;
				case kHIDFeatureReport:
					dataLength = (ptPreparsedData->reports[iR].featureBitCount + 7)/8;
					break;
				default:
					return kHIDInvalidReportTypeErr;
			}
			break;
		}
	}

	// If the reportID > 0, there must be 1 byte for reportID, so total report must be > 1.
	// (Would come into play if we had input report 3, but searched for ouput report 3
	// that didn't exist.)

	if (((reportID == 0) && (dataLength > 0)) || dataLength > 1)
	{
		iStatus = 0;
	}
	else
	{
		dataLength = 0;		// Ignore report that had id, but no data.
	}
	
	*reportLength = dataLength;

	return iStatus;
}

