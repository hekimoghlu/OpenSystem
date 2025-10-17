/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 16, 2024.
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
// kerneldiskrep - the kernel's own disk representation.
//
// This is a very special case.
// It's here primarily so we don't have to add special cases for the kernel
// all over the higher layers.
//
#ifndef _H_KERNELDISKREP
#define _H_KERNELDISKREP

#include "diskrep.h"

namespace Security {
namespace CodeSigning {


//
// A KernelDiskRep represents a (the) kernel on disk.
// It has no write support, so we can't sign the kernel,
// which is fine since we unconditionally trust it anyway.
//
class KernelDiskRep : public DiskRep {
public:
	KernelDiskRep();
	
	CFDataRef component(CodeDirectory::SpecialSlot slot);
	CFDataRef identification();
	std::string mainExecutablePath();
	CFURLRef copyCanonicalPath();
	size_t signingLimit();
	size_t execSegLimit(const Architecture *arch);
	std::string format();
	UnixPlusPlus::FileDesc &fd();
	
	std::string recommendedIdentifier(const SigningContext &ctx);
};


} // end namespace CodeSigning
} // end namespace Security

#endif // !_H_KERNELDISKREP
