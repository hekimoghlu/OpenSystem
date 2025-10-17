/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 3, 2023.
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
#ifndef _CFCLASS_H
#define _CFCLASS_H

#include <list>
#include <CoreFoundation/CFRuntime.h>
#include "threading.h"

namespace Security {

//
// CFClass
//
class CFClass : protected CFRuntimeClass
{
public:
    explicit CFClass(const char *name);

	CFTypeID typeID;

private:
	static void finalizeType(CFTypeRef cf) _NOEXCEPT;
    static Boolean equalType(CFTypeRef cf1, CFTypeRef cf2) _NOEXCEPT;
    static CFHashCode hashType(CFTypeRef cf) _NOEXCEPT;
	static CFStringRef copyFormattingDescType(CFTypeRef cf, CFDictionaryRef dict) _NOEXCEPT;
	static CFStringRef copyDebugDescType(CFTypeRef cf) _NOEXCEPT;
    static uint32_t refCountForType(intptr_t op, CFTypeRef cf) _NOEXCEPT;
    static uint32_t cleanupObject(intptr_t op, CFTypeRef cf, bool &zap);
};

} // end namespace Security

#endif
