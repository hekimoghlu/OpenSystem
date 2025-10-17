/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 1, 2024.
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
#ifndef webdavfs_webdav_utils_h
#define webdavfs_webdav_utils_h

/* Arrays of asctime-date day and month strs, rfc1123-date day and month strs, and rfc850-date day and month strs. */
static const char* kDayStrs[] = {
	"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
	"Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"};

static const char* kMonthStrs[] = {
	"January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December",
	"Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
	"jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"};

/* NOTE that these are ordered this way on purpose. */
static const char* kUSTimeZones[] = {"PST", "PDT", "MST", "MDT", "CST", "CDT", "EST", "EDT"};

static const uint8_t daysInMonth[16] = {0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31, 0, 0, 0};

typedef struct {
    SInt32 year;
    SInt8 month;
    SInt8 day;
    SInt8 hour;
    SInt8 minute;
    double second;
}Date;

const UInt8* CFGregorianDateCreateWithBytes(CFAllocatorRef alloc, const UInt8* bytes, CFIndex length, Date* date, CFTimeZoneRef* tz);

CFIndex CFGregorianDateCreateWithString(CFAllocatorRef alloc, CFStringRef str, Date* date, CFTimeZoneRef* tz);

Boolean IsLeapYear(SInt32 year);

Boolean DateIsValid(Date gdate);

/*
 * DateBytesToTime parses the RFC 850, RFC 1123, and asctime formatted
 * date/time bytes and returns time_t. If the parse fails, this function
 * returns a time_t set to -1.
 */
time_t DateBytesToTime(	
	const UInt8 *bytes,	/* -> pointer to bytes to parse */
	CFIndex length);	/* -> number of bytes to parse */

char* createUTF8CStringFromCFString(CFStringRef in_string);

/*
 * DateStringToTime parses the RFC 850, RFC 1123, and asctime formatted
 * date/time CFString and returns time_t. If the parse fails, this function
 * returns -1.
 */
time_t DateStringToTime(	/* <- time_t value; -1 if error */
		CFStringRef str);	/* -> CFString to parse */

#endif
