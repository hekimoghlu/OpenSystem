/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 22, 2025.
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
	File:		HIDLib.h

	Contains:	xxx put contents here xxx

	Version:	xxx put version here xxx

	Copyright:	ï¿½ 1999 by Apple Computer, Inc., all rights reserved.

	File Ownership:

		DRI:				xxx put dri here xxx

		Other Contact:		xxx put other contact here xxx

		Technology:			xxx put technology here xxx

	Writers:

		(BWS)	Brent Schorsch

	Change History (most recent first):

	  <USB1>	  3/5/99	BWS		first checked in
*/

#ifndef __HIDLib__
#define __HIDLib__

#include "HIDPriv.h"

#define	kShouldClearMem		true

/*------------------------------------------------------------------------------*/
/*																				*/
/* HID Library definitions														*/
/*																				*/
/*------------------------------------------------------------------------------*/

/* And now our extern procedures that are not external entry points in our shared library */

struct HIDReportDescriptor
{
	UInt8 *				descriptor;
	IOByteCount 		descriptorLength;
	UInt32				index;
	SInt32 *			collectionStack;
	SInt32				collectionNesting;
	HIDGlobalItems *	globalsStack;
	SInt32				globalsNesting;
	HIDItem				item;
	SInt32				firstUsageItem;
	SInt32				firstStringItem;
	SInt32				firstDesigItem;
	SInt32				parent;
	SInt32				sibling;
	HIDGlobalItems		globals;
	Boolean				haveUsageMin;
	Boolean				haveUsageMax;
	SInt32				rangeUsagePage;
	SInt32				usageMinimum;
	SInt32				usageMaximum;
	Boolean				haveStringMin;
	Boolean				haveStringMax;
	SInt32				stringMinimum;
	SInt32				stringMaximum;
	Boolean				haveDesigMin;
	Boolean				haveDesigMax;
	SInt32				desigMinimum;
	SInt32				desigMaximum;
};
typedef struct HIDReportDescriptor	HIDReportDescriptor;

/* And now our extern procedures that are not external entry points in our shared library */

extern OSStatus
HIDCountDescriptorItems	   (HIDReportDescriptor *	reportDescriptor,
							HIDPreparsedDataPtr 	preparsedData);

extern OSStatus
HIDNextItem				   (HIDReportDescriptor *	reportDescriptor);

extern OSStatus
HIDParseDescriptor		   (HIDReportDescriptor *	reportDescriptor,
							HIDPreparsedDataPtr 	preparsedData);

extern OSStatus
HIDProcessCollection	   (HIDReportDescriptor *	reportDescriptor,
							HIDPreparsedDataPtr 	preparsedData);

extern OSStatus
HIDProcessEndCollection	   (HIDReportDescriptor *	reportDescriptor,
							HIDPreparsedDataPtr 	preparsedData);

extern OSStatus
HIDProcessGlobalItem	   (HIDReportDescriptor *	reportDescriptor,
							HIDPreparsedDataPtr 	preparsedData);

extern OSStatus
HIDProcessLocalItem		   (HIDReportDescriptor *	reportDescriptor,
							HIDPreparsedDataPtr 	preparsedData);

extern OSStatus
HIDProcessMainItem		   (HIDReportDescriptor *	reportDescriptor,
							HIDPreparsedDataPtr 	preparsedData);

extern OSStatus
HIDProcessReportItem	   (HIDReportDescriptor *	reportDescriptor,
							HIDPreparsedDataPtr 	preparsedData);

#endif
