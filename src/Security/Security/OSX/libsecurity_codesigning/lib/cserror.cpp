/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 14, 2022.
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
// cserror.h - extended-diagnostics Code Signing errors
//
#include "cs.h"
#include <security_utilities/cfmunge.h>

namespace Security {
namespace CodeSigning {


//
// We need a nothrow destructor
//
CSError::~CSError() _NOEXCEPT
{ }


//
// Create and throw various forms of CSError
//
void CSError::throwMe(OSStatus rc)
{
	throw CSError(rc);
}

void CSError::throwMe(OSStatus rc, CFDictionaryRef dict)
{
	throw CSError(rc, dict);
}

void CSError::throwMe(OSStatus rc, CFStringRef key, CFTypeRef value)
{
	throw CSError(rc, cfmake<CFDictionaryRef>("{%O=%O}", key, value));
}


//
// Add a key/value pair to the dictionary
//
void CSError::augment(CFStringRef key, CFTypeRef value)
{
	mInfoDict.take(cfmake<CFDictionaryRef>("{+%O,%O=%O}", mInfoDict.get(), key, value));
}


//
// Convert exception-carried error information to CFError form
//
OSStatus CSError::cfError(CFErrorRef *errors) const
{
	if (errors)		// errors argument was specified
		*errors = CFErrorCreate(NULL, kCFErrorDomainOSStatus, this->osStatus(), this->infoDict());
	return this->osStatus();
}

OSStatus CSError::cfError(CFErrorRef *errors, OSStatus rc)
{
	if (errors)
		*errors = CFErrorCreate(NULL, kCFErrorDomainOSStatus, rc, NULL);
	return rc;
}


}	// CodeSigning
}	// Security
