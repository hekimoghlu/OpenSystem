/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 8, 2022.
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
// detachedrep - prefix diskrep representing a detached signature stored in a file
//
#ifndef _H_DETACHEDREP
#define _H_DETACHEDREP

#include "diskrep.h"
#include "sigblob.h"

namespace Security {
namespace CodeSigning {


//
// We use a DetachedRep to interpose (filter) the genuine DiskRep representing
// the code on disk, *if* a detached signature was set on this object. In this
// situation, mRep will point to a (2 element) chain of DiskReps.
//
// This is a neat way of dealing with the (unusual) detached-signature case
// without disturbing things unduly. Consider DetachedDiskRep to be closely
// married to SecStaticCode; it's unlikely to work right if you use it elsewhere.
//
// Note that there's no *writing* code here. Writing detached signatures is handled
// specially in the signing code.
//
class DetachedRep : public FilterRep {
public:
	DetachedRep(CFDataRef sig, DiskRep *orig, const std::string &source); // SuperBlob of all architectures
	DetachedRep(CFDataRef sig, CFDataRef gsig, DiskRep *orig, const std::string &source); // one architecture + globals
	
	CFDataRef component(CodeDirectory::SpecialSlot slot);
	
	bool fullSignature() const { return mFull; }
	const std::string &source() const { return mSource; }

private:
	CFCopyRef<CFDataRef> mSig, mGSig;
	bool mFull;								// full detached signature (explicitly given)
	const EmbeddedSignatureBlob *mArch;		// current architecture; points into mSignature
	const EmbeddedSignatureBlob *mGlobal;	// shared elements; points into mSignature
	std::string mSource;					// source description (readable)
};


} // end namespace CodeSigning
} // end namespace Security

#endif // !_H_DETACHEDREP
