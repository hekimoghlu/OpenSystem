/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 26, 2025.
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
// machorep - DiskRep mix-in for handling Mach-O main executables
//
#ifndef _H_MACHOREP
#define _H_MACHOREP

#include "singlediskrep.h"
#include "sigblob.h"
#include <security_utilities/unix++.h>
#include <security_utilities/macho++.h>

namespace Security {
namespace CodeSigning {


//
// MachORep is a DiskRep class that supports code signatures
// directly embedded in Mach-O binary files.
//
// It does not have write support (for writing signatures);
// writing multi-architecture binaries is complicated enough
// that it's driven directly from the signing code, with no
// abstractions to get in the way.
//
class MachORep : public SingleDiskRep, public EditableDiskRep {
public:
	MachORep(const char *path, const Context *ctx = NULL);
	virtual ~MachORep();
	
	CFDataRef component(CodeDirectory::SpecialSlot slot);
	RawComponentMap createRawComponents();
	CFDataRef identification();
	Universal *mainExecutableImage();
	void prepareForSigning(SigningContext &context);
	size_t signingBase();
	size_t signingLimit();
	size_t execSegBase(const Architecture *arch);
	size_t execSegLimit(const Architecture *arch);
	uint32_t platformType(const Architecture *arch);
	std::string format();
    CFDictionaryRef copyDiskRepInformation();

	std::string recommendedIdentifier(const SigningContext &ctx);
	std::string explicitIdentifier();
	const Requirements *defaultRequirements(const Architecture *arch, const SigningContext &ctx);
	size_t pageSize(const SigningContext &ctx);

	void strictValidate(const CodeDirectory* cd, const ToleratedErrors& tolerated, SecCSFlags flags);
	
	void flush();		// flush cache

	static bool candidate(UnixPlusPlus::FileDesc &fd);
	void registerStapledTicket();
	CFDataRef copyStapledTicket();
	
public:
	static CFDataRef identificationFor(MachO *macho);
	
public:
	DiskRep::Writer *writer();
	class Writer;
	friend class Writer;
	
protected:
	CFDataRef embeddedComponent(CodeDirectory::SpecialSlot slot);
	CFDataRef infoPlist();
	Requirement *libraryRequirements(const Architecture *arch, const SigningContext &ctx);

private:
	static bool needsExecSeg(const MachO& macho);
	EmbeddedSignatureBlob *signingData();

	Universal *mExecutable;	// cached Mach-O/Universal reference to mainExecutablePath()
	mutable EmbeddedSignatureBlob *mSigningData; // cached signing data from current architecture
};


//
// The write side of a MachORep.
// This is purposely dysfunctional; Mach-O signatures are written
// by code in signerutils, not by DiskRep::Writers.
//
class MachORep::Writer : public SingleDiskRep::Writer {
	friend class FileDiskRep;
public:
	Writer(MachORep *r) : SingleDiskRep::Writer(r, writerNoGlobal) { }
	void component(CodeDirectory::SpecialSlot slot, CFDataRef data);
};


} // end namespace CodeSigning
} // end namespace Security

#endif // !_H_MACHOREP
