/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 26, 2025.
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

#ifndef __CPLUSPLUS_UTILS__
#define __CPLUSPLUS_UTILS__

#include <string>
#include <CoreFoundation/CoreFoundation.h>

std::string StringFromCFString(CFStringRef theString);
CFStringRef CFStringFromString(std::string theString) CF_RETURNS_RETAINED;

// class to automatically manage the lifetime of a CFObject

class CFTypeRefHolder
{
private:
	CFTypeRef mTypeRef;

public:
	CFTypeRefHolder(CFTypeRef typeRef) : mTypeRef(typeRef) {}
	virtual ~CFTypeRefHolder();
	
	void Set(CFTypeRef typeRef); // replace the value in the holder with another -- releases the current value
	CFTypeRef Get() {return mTypeRef;}
};



#endif
