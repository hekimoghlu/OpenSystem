/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 18, 2025.
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
// diskimagerep - DiskRep representing a single read-only compressed disk image file
//
#ifndef _H_DISKIMAGEREP
#define _H_DISKIMAGEREP

#include "singlediskrep.h"
#include "sigblob.h"
#include <DiskImages/DiskImages.h>
#undef check	// sadness is having to live with C #defines of this kind...
#include <security_utilities/unix++.h>

namespace Security {
namespace CodeSigning {


//
// DiskImageRep implements a single read-only compressed disk image file.
//
class DiskImageRep : public SingleDiskRep {
public:
	DiskImageRep(const char *path);
	virtual ~DiskImageRep();
	
	CFDataRef identification();
	CFDataRef component(CodeDirectory::SpecialSlot slot);
	size_t signingLimit();
	void strictValidate(const CodeDirectory* cd, const ToleratedErrors& tolerated, SecCSFlags flags);
	std::string format();
	void prepareForSigning(SigningContext& state);
	
	static bool candidate(UnixPlusPlus::FileDesc &fd);
	void registerStapledTicket();
	CFDataRef copyStapledTicket();

public:
	static CFDataRef identificationFor(MachO *macho);
	
public:
	DiskRep::Writer *writer();
	class Writer;
	friend class Writer;

private:
	void setup();
	static bool readHeader(UnixPlusPlus::FileDesc& fd, UDIFFileHeader& header);

private:
	UDIFFileHeader mHeader;						// disk image header (all fields NBO)
	size_t mEndOfDataOffset;					// end of payload data (data fork + XML)
	size_t mHeaderOffset;						// trailing header offset
	const EmbeddedSignatureBlob *mSigningData;	// pointer to signature SuperBlob (malloc'd memory during setup)
};


//
// The write side of a FileDiskRep
//
class DiskImageRep::Writer : public SingleDiskRep::Writer, private EmbeddedSignatureBlob::Maker {
	friend class FileDiskRep;
public:
	Writer(DiskImageRep *r) : SingleDiskRep::Writer(r, writerNoGlobal), rep(r), mSigningData(NULL) { }
	void component(CodeDirectory::SpecialSlot slot, CFDataRef data);
	void flush();
	void addDiscretionary(CodeDirectory::Builder &builder);
	
private:
	DiskImageRep *rep;
	EmbeddedSignatureBlob *mSigningData;
};


} // end namespace CodeSigning
} // end namespace Security

#endif // !_H_DISKIMAGEREP
