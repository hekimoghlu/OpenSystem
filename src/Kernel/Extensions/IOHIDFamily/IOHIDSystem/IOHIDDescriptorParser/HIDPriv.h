/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 5, 2025.
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
#ifndef __HIDPriv__
#define __HIDPriv__

/*
	File:		HIDPriv.i

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

		 <5>	 11/1/99	BWS		[2405720]  We need a better check for 'bit padding' items,
									rather than just is constant. We will check to make sure the
									item is constant, and has no usage, or zero usage. This means we
									need to pass an additional parameter to some internal functions
		 <4>	  4/7/99	BWS		Add flags to report items (for reverse)
		 <3>	 3/19/99	BWS		Build stub library
		 <2>	 3/17/99	BWS		[2314839]  Add flags field to HIDPreparsedData, is set in
									HIDOpenReportDescriptor
		 <1>	  3/5/99	BWS		first checked in
*/

#include "IOHIDDescriptorParserPrivate.h"

#if __has_include(<os/overflow.h>)
#include <os/overflow.h>
#elif __has_builtin(__builtin_add_overflow)
#define os_add_overflow(a, b, c) __builtin_add_overflow(a, b, c)
#define os_sub_overflow(a, b, c) __builtin_sub_overflow(a, b, c)
#define os_mul_overflow(a, b, c) __builtin_mul_overflow(a, b, c)
#else
#error unsupported compiler
#endif

/*------------------------------------------------------------------------------*/
/*																				*/
/* HIDLibrary private defs														*/
/*																				*/
/*------------------------------------------------------------------------------*/

extern 
OSStatus
HIDCheckReport			   (HIDReportType 			reportType,
							HIDPreparsedDataRef		preparsedDataRef,
							HIDReportItem *			reportItem,
							void * 					report,
							IOByteCount				reportLength);


extern 
OSStatus
HIDGetData				   (void *					report,
							IOByteCount				reportLength,
							UInt32					start,
							UInt32					size,
							SInt32 *				value,
							Boolean 				signExtend);

extern 
OSStatus
HIDPostProcessRIValue 	   (HIDReportItem *			reportItem,
							SInt32 *				value);

extern 
OSStatus
HIDPreProcessRIValue  	   (HIDReportItem *	 		reportItem,
							SInt32 *				value);
							
extern
Boolean
HIDHasUsage				   (HIDPreparsedDataRef		preparsedDataRef,
							HIDReportItem *			reportItem,
							HIDUsage				usagePage,
							HIDUsage				usage,
							UInt32 *				usageIndex,
							UInt32 *				count);

extern
Boolean
HIDIsButton				   (HIDReportItem *			reportItem,
							HIDPreparsedDataRef		preparsedDataRef);

extern
Boolean
HIDIsVariable			   (HIDReportItem *			reportItem,
							HIDPreparsedDataRef		preparsedDataRef);

extern 
OSStatus
HIDPutData				   (void *					report,
							IOByteCount				reportLength,
							UInt32					start,
							UInt32					size,
							SInt32 					value);

extern 
OSStatus
HIDScaleUsageValueIn	   (HIDReportItem *			reportItem,
							UInt32 					value,
							SInt32 *				scaledValue);

extern 
OSStatus
HIDScaleUsageValueOut	   (HIDReportItem *			reportItem,
							UInt32 					value,
							SInt32 *				scaledValue);

extern 
void
HIDUsageAndPageFromIndex   (HIDPreparsedDataRef		preparsedDataRef,
							HIDReportItem *			reportItem,
							UInt32 					index,
							HIDUsageAndPage *		usageAndPage);

extern
Boolean
HIDUsageInRange			   (HIDP_UsageItem *		usageItem, 
							HIDUsage				usagePage,
							HIDUsage				usage);

#endif
