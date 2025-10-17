/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 2, 2024.
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
//
// Manage the Tower of Babel of CSSM dates and times.
//
#ifndef _H_CSSMDATES
#define _H_CSSMDATES

#include <security_utilities/utilities.h>
#include <security_cdsa_utilities/cssmdata.h>
#include <CoreFoundation/CFDate.h>


namespace Security {


//
// A PodWrapper for CSSM_DATE
//
class CssmDate : public PodWrapper<CssmDate, CSSM_DATE> {
public:
    CssmDate() { }
    CssmDate(const char *y, const char *m, const char *d);
    CssmDate(int y, int m, int d);
    
    const char *years() const	{ return reinterpret_cast<const char *>(Year); }
    const char *months() const	{ return reinterpret_cast<const char *>(Month); }
    const char *days() const	{ return reinterpret_cast<const char *>(Day); }
    char *years()				{ return reinterpret_cast<char *>(Year); }
    char *months()				{ return reinterpret_cast<char *>(Month); }
    char *days()				{ return reinterpret_cast<char *>(Day); }
    
    int year() const;
    int month() const;
    int day() const;
    
private:
    static void assign(char *dest, int width, const char *src);
};

inline bool operator == (const CSSM_DATE &d1, const CSSM_DATE &d2)
{ return !memcmp(&d1, &d2, sizeof(d1)); }

inline bool operator != (const CSSM_DATE &d1, const CSSM_DATE &d2)
{ return !memcmp(&d1, &d2, sizeof(d1)); }


//
// Yet another CSSM date/time format is CSSM_TIMESTRING. This is
// defined as "char *", just so you can't use the type system
// to keep things coordinated, so we can't really PodWrap it the usual way.
// What *were* they thinking?
// The format is allegedly "yyyymmddhhmmss", and the standard says
// nothing about trailing null characters.
//


//
// A unified date-and-time object.
// This is based on CFDate objects and converts to various CSSM
// inspired formats.
//
class CssmUniformDate {
public:
    CssmUniformDate()	{ }
    
    // convert to/from CFDateRef
    CssmUniformDate(CFDateRef ref);
    operator CFDateRef() const;
    
	// convert to/from CFAbsoluteTime
	CssmUniformDate(CFAbsoluteTime ct) : mTime(ct) { }
	operator CFAbsoluteTime() const { return mTime; }
	
    // convert to/from CSSM_DATE
    CssmUniformDate(const CssmDate &src);
    operator CssmDate () const;
    
    // convert to/from DATA format (1999-06-30_15:05:39 form)
    CssmUniformDate(const CSSM_DATA &src);
    void convertTo(CssmOwnedData &data) const;
    
    // convert to/from CSSM_TIMESTRING format (19990630150539)
    CssmUniformDate(const char *src);
    void convertTo(char *dest, size_t length) const;

    // native comparisons
    bool operator < (const CssmUniformDate &other) const	{ return mTime < other.mTime; }
    bool operator == (const CssmUniformDate &other) const	{ return mTime == other.mTime; }
    bool operator > (const CssmUniformDate &other) const	{ return mTime > other.mTime; }
    bool operator <= (const CssmUniformDate &other) const	{ return mTime <= other.mTime; }
    bool operator >= (const CssmUniformDate &other) const	{ return mTime >= other.mTime; }
    bool operator != (const CssmUniformDate &other) const	{ return mTime != other.mTime; }

private:
    void setFromString(const char *src, const char *format, size_t fieldLength);

private:
    CFAbsoluteTime mTime;
};


} // end namespace Security

#endif //_H_CSSMDATES
