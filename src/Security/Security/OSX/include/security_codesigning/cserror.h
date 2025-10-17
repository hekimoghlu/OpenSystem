/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 24, 2023.
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
#ifndef _H_CSERRORS
#define _H_CSERRORS

#include <security_utilities/cfutilities.h>
#include <security_utilities/debugging.h>


namespace Security {
namespace CodeSigning {


//
// Special tailored exceptions to transmit additional error information
//
class CSError : public MacOSError {
public:
	CSError(OSStatus rc) : MacOSError(rc) { }
	CSError(OSStatus rc, CFDictionaryRef dict) : MacOSError(rc), mInfoDict(dict) { } // takes dict
	~CSError() _NOEXCEPT;
	
    static void throwMe(OSStatus rc) __attribute__((noreturn));
	static void throwMe(OSStatus rc, CFDictionaryRef info) __attribute__ ((noreturn)); // takes dict
    static void throwMe(OSStatus rc, CFStringRef key, CFTypeRef value) __attribute__((noreturn));

	void augment(CFStringRef key, CFTypeRef value);

	CFDictionaryRef infoDict() const { return mInfoDict; }
	
public:
	OSStatus cfError(CFErrorRef *errors) const;
	static OSStatus cfError(CFErrorRef *errors, OSStatus rc);
	
private:
	CFRef<CFDictionaryRef> mInfoDict;
};


}	// CodeSigning
}	// Security

#endif //_H_CSERRORS
