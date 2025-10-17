/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 14, 2023.
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
 *  cssmdatetime.h -- defines for the CSSM date and time utilities for the Mac
 */
#ifndef _SECURITY_CSSMDATETIME_H_
#define _SECURITY_CSSMDATETIME_H_

#include <Security/cssm.h>
#include <CoreFoundation/CFDate.h>

namespace Security
{

namespace CSSMDateTimeUtils
{

// Get the current time.
extern void GetCurrentMacLongDateTime(sint64 &outMacDate);

extern void TimeStringToMacSeconds(const CSSM_DATA &inUTCTime, uint32 &ioMacDate);
extern void TimeStringToMacLongDateTime(const CSSM_DATA &inUTCTime, sint64 &outMacDate);

// Length of inLength is an input parameter and must be 14 or 16.
// The outData parameter must point to a buffer of at least inLength bytes.
extern void MacSecondsToTimeString(uint32 inMacDate, uint32 inLength, void *outData, uint32 outLength);
extern void MacLongDateTimeToTimeString(const sint64 &inMacDate,
                                        uint32 inLength, void *outData, uint32 outLength);

// outCssmDate must be a pointer to a 16 byte buffer.
extern void CFDateToCssmDate(CFDateRef date, char *outCssmDate, uint32 outCssmDateLen);

// cssmDate must be a pointer to a 16 byte buffer formatted as follows: "YYYYMMDDhhmmssZ\0".
extern void CssmDateToCFDate(const char *cssmDate, CFDateRef *outCFDate);

// cssmDate must be a pointer to a string buffer formatted as follows: "[YY]YYMMDDhhmmssZ[\0]".
// handles UTC (2-digit year) or generalized (4-digit year) date strings; terminated or not.
// also handles localized time formats: "[YY]MMDDhhmmssThhmm" (where T=[+,-]).
extern int CssmDateStringToCFDate(const char *cssmDate, unsigned int len, CFDateRef *outCFDate);

} // end namespace CSSMDateTimeUtils

} // end namespace Security

#endif // !_SECURITY_CSSMDATETIME_H_
