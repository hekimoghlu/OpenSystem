/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 6, 2024.
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
#include "diskimagerep.h"
#include "notarization.h"
#include "sigblob.h"
#include "CodeSigner.h"
#include <security_utilities/endian.h>
#include <algorithm>


namespace Security {
namespace CodeSigning {

using Security::n2h;
using Security::h2n;
using namespace UnixPlusPlus;


static const int32_t udifVersion = 4;		// supported image file version
	

//
// Temporary hack to imply a fUDIFCryptosigFieldsset at the start of the "reserved" area of an UDIF header
//
bool DiskImageRep::readHeader(FileDesc& fd, UDIFFileHeader& header)
{
	// the UDIF "header" is in fact the last 512 bytes of the file, with no particular alignment
	static const size_t headerLength = sizeof(header);
	size_t length = fd.fileSize();
	if (length < sizeof(UDIFFileHeader) + sizeof(BlobCore))
		return false;
	size_t headerOffset = length - sizeof(UDIFFileHeader);
	if (fd.read(&header, headerLength, headerOffset) != headerLength)
		return false;
	if (n2h(header.fUDIFSignature) != kUDIFSignature)
		return false;
	if (n2h(header.fUDIFVersion) != udifVersion)	// current as of this writing
		return false;
	
	return true;
}


//
// Object management.
//
DiskImageRep::DiskImageRep(const char *path)
	: SingleDiskRep(path), mSigningData(NULL)
{
	this->setup();
}

DiskImageRep::~DiskImageRep()
{
	free((void*)mSigningData);
}

void DiskImageRep::setup()
{
	free((void*)mSigningData);
	mSigningData = NULL;
	
	// the UDIF "header" is in fact the last 512 bytes of the file, with no particular alignment
	if (!readHeader(fd(), this->mHeader))
		UnixError::throwMe(errSecCSBadDiskImageFormat);

	mHeaderOffset = fd().fileSize() - sizeof(UDIFFileHeader);
	size_t signatureOffset = size_t(n2h(this->mHeader.fUDIFCodeSignOffset));
	size_t signatureLength = size_t(n2h(this->mHeader.fUDIFCodeSignLength));
	this->mHeader.fUDIFCodeSignLength = 0;		// blind length (signature covers header)
	if (signatureOffset == 0) {
		mEndOfDataOffset = mHeaderOffset;
		mHeader.fUDIFCodeSignOffset = h2n(mHeaderOffset);
		return;		// unsigned, header prepared for possible signing
	} else {
		mEndOfDataOffset = signatureOffset;
	}
	
	// read the signature superblob
	const size_t frameLength = mHeaderOffset - signatureOffset;		// room to following header
	if (EmbeddedSignatureBlob* blob = EmbeddedSignatureBlob::readBlob(fd(), signatureOffset, frameLength)) {
		if (blob->length() != frameLength
				|| frameLength != signatureLength
				|| !blob->strictValidateBlob(frameLength)) {
			free(blob);
			MacOSError::throwMe(errSecCSBadDiskImageFormat);
		}
		mSigningData = blob;
	}
}
	
	
//
// The default binary identification of a SingleDiskRep is the (SHA-1) hash
// of the entire file itself.
//
CFDataRef DiskImageRep::identification()
{
	SHA1 hash;		// not security sensitive
	hash(&mHeader, sizeof(mHeader));
	SHA1::Digest digest;
	hash.finish(digest);
	return makeCFData(digest, sizeof(digest));
}


//
// Sniffer function for UDIF disk image files.
// This just looks for the trailing "header" and its magic number.
//
bool DiskImageRep::candidate(FileDesc &fd)
{
	UDIFFileHeader header;
	return readHeader(fd, header) == true;
}


//
// Signing limit is the start of the (trailing) signature
//
size_t DiskImageRep::signingLimit()
{
	return mEndOfDataOffset;
}

void DiskImageRep::strictValidate(const CodeDirectory* cd, const ToleratedErrors& tolerated, SecCSFlags flags)
{
    DiskRep::strictValidate(cd, tolerated, flags);

    if (cd) {
        size_t cd_limit = cd->signingLimit();
        size_t dr_limit = signingLimit();
        if (cd_limit != dr_limit &&         // must cover exactly the entire data
            cd_limit != fd().fileSize())    // or, for legacy detached sigs, the entire file
            MacOSError::throwMe(errSecCSSignatureInvalid);
    }
}


//
// Retrieve a component from the executable.
// Our mCache has mapped the entire file, so we just fish the contents out of
// the mapped area as needed.
//
CFDataRef DiskImageRep::component(CodeDirectory::SpecialSlot slot)
{
	switch (slot) {
	case cdRepSpecificSlot:
		return makeCFData(&mHeader, sizeof(mHeader));
	default:
		return mSigningData ? mSigningData->component(slot) : NULL;
	}
}


//
// Provide a (vaguely) human readable characterization of this code
//
string DiskImageRep::format()
{
	return "disk image";
}
	
void DiskImageRep::prepareForSigning(SigningContext& context)
{
	// default to SHA256 unconditionally - we have no legacy issues to worry about
	if (context.digestAlgorithms().empty())
		context.setDigestAlgorithm(kSecCodeSignatureHashSHA256);
}


//
// DiskImageRep::Writers
//
DiskRep::Writer *DiskImageRep::writer()
{
	return new Writer(this);
}


//
// Write a component.
//
void DiskImageRep::Writer::component(CodeDirectory::SpecialSlot slot, CFDataRef data)
{
	assert(slot != cdRepSpecificSlot);
	EmbeddedSignatureBlob::Maker::component(slot, data);
}


//
// Append the superblob we built to the cache file.
//
void DiskImageRep::Writer::flush()
{
	free((void*)mSigningData);		// ditch previous blob just in case
	mSigningData = Maker::make();	// assemble new signature SuperBlob
	
	// write signature superblob
	size_t location = rep->mEndOfDataOffset;
	assert(location);
	fd().seek(location);
	fd().writeAll(*mSigningData);	// write signature
	
	// now (re)write disk image header after it
	UDIFFileHeader fullHeader = rep->mHeader;
	fullHeader.fUDIFCodeSignOffset = h2n(location);
	fullHeader.fUDIFCodeSignLength = h2n(mSigningData->length());
	fd().writeAll(&fullHeader, sizeof(rep->mHeader));
    fd().truncate(fd().position());
}


//
// Discretionary manipulations
//
void DiskImageRep::Writer::addDiscretionary(CodeDirectory::Builder &builder)
{
}

void DiskImageRep::registerStapledTicket()
{
	CFRef<CFDataRef> data = copyStapledTicket();
	registerStapledTicketWithSystem(data);
}

CFDataRef DiskImageRep::copyStapledTicket()
{
	CFRef<CFDataRef> data = NULL;
	if (mSigningData) {
		data.take(mSigningData->component(cdTicketSlot));
	}
	return data.yield();
}


} // end namespace CodeSigning
} // end namespace Security
