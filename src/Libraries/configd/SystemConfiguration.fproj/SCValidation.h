/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 30, 2024.
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
#ifndef _SCVALIDATION_H
#define _SCVALIDATION_H

#include <sys/cdefs.h>
#include <CoreFoundation/CoreFoundation.h>

__BEGIN_DECLS

static __inline__ CFTypeRef
isA_CFType(CFTypeRef obj, CFTypeID type)
{
	if (obj == NULL)
		return (NULL);

	if (CFGetTypeID(obj) != type)
		return (NULL);

	return (obj);
}

static __inline__ CFTypeRef
isA_CFArray(CFTypeRef obj)
{
	return (isA_CFType(obj, CFArrayGetTypeID()));
}

static __inline__ CFTypeRef
isA_CFBoolean(CFTypeRef obj)
{
	return (isA_CFType(obj, CFBooleanGetTypeID()));
}

static __inline__ CFTypeRef
isA_CFData(CFTypeRef obj)
{
	return (isA_CFType(obj, CFDataGetTypeID()));
}

static __inline__ CFTypeRef
isA_CFDate(CFTypeRef obj)
{
	return (isA_CFType(obj, CFDateGetTypeID()));
}

static __inline__ CFTypeRef
isA_CFDictionary(CFTypeRef obj)
{
	return (isA_CFType(obj, CFDictionaryGetTypeID()));
}

static __inline__ CFTypeRef
isA_CFNumber(CFTypeRef obj)
{
	return (isA_CFType(obj, CFNumberGetTypeID()));
}

static __inline__ CFTypeRef
isA_CFPropertyList(CFTypeRef obj)
{
	CFTypeID	type;

	if (obj == NULL)
		return (NULL);

	type = CFGetTypeID(obj);
	if (type == CFArrayGetTypeID()		||
	    type == CFBooleanGetTypeID()	||
	    type == CFDataGetTypeID()		||
	    type == CFDateGetTypeID()		||
	    type == CFDictionaryGetTypeID()	||
	    type == CFNumberGetTypeID()		||
	    type == CFStringGetTypeID())
		return (obj);

	return (NULL);
}


static __inline__ CFTypeRef
isA_CFString(CFTypeRef obj)
{
	return (isA_CFType(obj, CFStringGetTypeID()));
}


Boolean
_SC_stringIsValidDNSName	(const char *name);


Boolean
_SC_CFStringIsValidDNSName	(CFStringRef name);


Boolean
_SC_CFStringIsValidNetBIOSName	(CFStringRef name);


__END_DECLS

#endif /* _SCVALIDATION_H */

