/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 1, 2021.
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
// cskernel - Kernel implementation of the Code Signing Host Interface
//
#ifndef _H_CSKERNEL
#define _H_CSKERNEL

#include "Code.h"
#include "StaticCode.h"
#include <security_utilities/utilities.h>

namespace Security {
namespace CodeSigning {


class ProcessCode;


//
// The nominal StaticCode representing the kernel on disk.
// This is barely used, since we don't validate the kernel (it's the root of trust)
// and we don't activate new kernels at runtime.
//
class KernelStaticCode : public SecStaticCode {
public:
	KernelStaticCode();

private:
};


//
// A SecCode that represents the system's running kernel.
// We usually only have one of those in the system at one time. :-)
//
class KernelCode : public SecCode {
public:
	KernelCode();

	SecCode *locateGuest(CFDictionaryRef attributes);
	SecStaticCode *identifyGuest(SecCode *guest, CFDataRef *cdhash);
	SecCodeStatus getGuestStatus(SecCode *guest);
	void changeGuestStatus(SecCode *guest, SecCodeStatusOperation operation, CFDictionaryRef arguments);
	void guestMatchesLightweightCodeRequirement(SecCode *guest, const Requirement* lwcr);
	
	static KernelCode *active()		{ return globals().code; }
	
public:
	struct Globals {
		Globals();
		SecPointer<KernelCode> code;
		SecPointer<KernelStaticCode> staticCode;
	};
	static ModuleNexus<Globals> globals;

protected:
	void identify();
	void csops(ProcessCode *proc, unsigned int op, void *addr = NULL, size_t length = 0);
};


} // end namespace CodeSigning
} // end namespace Security

#endif // !_H_CSKERNEL
