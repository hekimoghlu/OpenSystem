/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 5, 2022.
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
// CoreFoundation building and parsing functions
//
#ifndef _H_CFMUNGE
#define _H_CFMUNGE

#include <security_utilities/cfutilities.h>
#include <CoreFoundation/CoreFoundation.h>
#include <cstdarg>

namespace Security {


//
// Common interface to Mungers.
// A CFMunge provides a one-pass, non-resettable scan through a format string,
// performing various actions on the way.
//
class CFMunge {
public:
    // Initialize a CFMunge. We start out with the default CFAllocator, and
    // we do not throw errors.
    // CFMunge consumes the va_list, the caller should call va_copy if necessary.
    CFMunge(const char *fmt, va_list *args)
        : format(fmt), args(args), allocator(NULL), error(errSecSuccess) { }

protected:
	char next();
	bool next(char c);
	
	bool parameter();
	
protected:
	const char *format;
	va_list *args;
	CFAllocatorRef allocator;
	OSStatus error;
};


//
// A CFMake is a CFMunge for making CF data structures.
//
class CFMake : public CFMunge {
public:
	CFMake(const char *fmt, va_list *args) : CFMunge(fmt, args) { }
	
	CFTypeRef make();
	CFDictionaryRef addto(CFMutableDictionaryRef dict);
	
protected:
	CFTypeRef makedictionary();
	CFTypeRef makearray();
	CFTypeRef makenumber();
	CFTypeRef makestring();
	CFTypeRef makeformat();
	CFTypeRef makespecial();

	CFDictionaryRef add(CFMutableDictionaryRef dict);
};


//
// Make a CF object following a general recipe
//
CFTypeRef cfmake(const char *format, ...);
CFTypeRef vcfmake(const char *format, va_list *args);

template <class CFType>
CFType cfmake(const char *format, ...)
{
	va_list args;
	va_start(args, format);
	CFType result = CFType(vcfmake(format, &args));
	va_end(args);
	return result;
}

CFDictionaryRef cfadd(CFMutableDictionaryRef dict, const char *format, ...);


//
// Parse out parts of a CF object following a general recipe.
// Cfscan returns false on error; cfget throws.
//
bool cfscan(CFTypeRef source, const char *format, ...);
bool vcfscan(CFTypeRef source, const char *format, va_list *args);

CFTypeRef cfget(CFTypeRef source, const char *format, ...);
CFTypeRef vcfget(CFTypeRef source, const char *format, va_list *args);

template <class CFType>
CFType cfget(CFTypeRef source, const char *format, ...)
{
	va_list args;
	va_start(args, format);
	CFType result = CFType(vcfget(source, format, &args));
	va_end(args);
	return (result && CFTraits<CFType>::check(result)) ? result : NULL;
}

template <class CFType>
class CFTemp : public CFRef<CFType> {
public:
	CFTemp(const char *format, ...)
	{
		va_list args;
		va_start(args, format);
		this->take(CFType(vcfmake(format, &args)));
		va_end(args);
	}
};


}	// end namespace Security

#endif //_H_CFMUNGE
