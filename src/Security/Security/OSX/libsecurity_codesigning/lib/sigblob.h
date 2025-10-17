/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 28, 2022.
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
// sigblob - signature (Super)Blob types
//
#ifndef _H_SIGBLOB
#define _H_SIGBLOB

#include "codedirectory.h"
#include <security_utilities/superblob.h>
#include <CoreFoundation/CFData.h>

namespace Security {
namespace CodeSigning {


//
// An EmbeddedSignatureBlob is a SuperBlob indexed by component slot number.
// This is what we embed in Mach-O images. It is also what we use for detached
// signatures for non-Mach-O binaries.
//
class EmbeddedSignatureBlob : public SuperBlobCore<EmbeddedSignatureBlob, 0xfade0cc0, uint32_t> {
	typedef SuperBlobCore<EmbeddedSignatureBlob, 0xfade0cc0, uint32_t> _Core;
public:
	static CFDataRef blobData(CodeDirectory::SpecialSlot slot, BlobCore const *blob);
	CFDataRef component(CodeDirectory::SpecialSlot slot) const;
	
	class Maker : public _Core::Maker {
	public:
		void component(CodeDirectory::SpecialSlot type, CFDataRef data);
	};
};


//
// A DetachedSignatureBlob collects multiple architectures' worth of
// EmbeddedSignatureBlobs into one, well, Super-SuperBlob.
// This is what we use for Mach-O detached signatures.
//
typedef SuperBlob<0xfade0cc1> DetachedSignatureBlob;	// indexed by main architecture


//
// The linkers produces a superblob of dependency records from its dylib inputs
//
typedef SuperBlob<0xfade0c05> LibraryDependencyBlob; // indexed sequentially from 0


//
// An entitlement blob is used for embedding entitlement configuration data
//
class EntitlementBlob : public Blob<EntitlementBlob, 0xfade7171> {
public:
	CFDictionaryRef entitlements() const;
};

//
// Similar, but in DER representation.
//
class EntitlementDERBlob : public Blob<EntitlementDERBlob, kSecCodeMagicEntitlementDER> {
public:
	static EntitlementDERBlob *alloc(size_t length);

	uint8_t *der() { return data; }
	const uint8_t *der() const { return data; }
	size_t derLength() const { return BlobCore::length() - sizeof(BlobCore); }
private:
	uint8_t data[0];
};

class LaunchConstraintBlob : public Blob<LaunchConstraintBlob, kSecCodeMagicLaunchConstraint> {
public:
	static LaunchConstraintBlob *alloc(size_t length);

	uint8_t *der() { return data; }
	const uint8_t *der() const { return data; }
	size_t derLength() const { return BlobCore::length() - sizeof(BlobCore); }
private:
	uint8_t data[0];
};

} // end namespace CodeSigning
} // end namespace Security

#endif // !_H_SIGBLOB
