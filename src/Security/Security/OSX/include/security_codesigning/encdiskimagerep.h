/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 17, 2022.
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
#ifndef _H_ENCDISKIMAGEREP
#define _H_ENCDISKIMAGEREP

#include "singlediskrep.h"
#include "sigblob.h"
#include <DiskImages/DiskImages.h>
#undef check	// sadness is having to live with C #defines of this kind...
#include <security_utilities/unix++.h>

namespace Security {
namespace CodeSigning {

/// The serialized format of an encrypted disk image header, taken from CEncryptedEncoding.h
/// The overall format can be roughly described as:
///		__Encrypted_Header_V2
///		AuthTable
///			uint32_t - count of table entries
///			AuthTableEntry
///			...
///			AuthTableEntry
///		Authentication body - arbitrary data
///			NOTE: disk images are created with a large amount of padding here (~100kB), which prevents
///			the need to push out the actual data when adding small bits of data.  All data here is referenced by
///			offsets within the AuthTableEntries.
///		Start of disk image from dataForkOffset
struct __Encrypted_Header_V2 {
	uint32_t				signature1;
	uint32_t				signature2;
	uint32_t				version;

	uint32_t				encryptionIVSize;
	CSSM_ENCRYPT_MODE		encryptionMode;
	CSSM_ALGORITHMS			encryptionAlgorithm;
	uint32_t				encryptionKeySizeInBits;
	CSSM_ALGORITHMS			prngAlgorithm;
	uint32_t				prngKeySizeInBits;

	CFUUIDBytes				uuid;

	uint32_t				dataBlockSize;
	di_filepos_t			dataForkSize;
	di_filepos_t			dataForkStartOffset;
} __attribute__((packed));

struct __AuthTableEntry {
	uint32_t				mechanism;
	di_filepos_t			offset;
	di_filepos_t			length;
} __attribute__((packed));

/// Represents a single authentication table entry, inside an AuthTable, from an encrypted disk image header.
class AuthTableEntry {
public:
	AuthTableEntry(UnixPlusPlus::FileDesc &fd);
	AuthTableEntry(uint32_t mechanism, uint64_t offset, uint64_t length);
	~AuthTableEntry();

	void loadData(UnixPlusPlus::FileDesc &fd);
	void setOffset(uint64_t newOffset);
	void setData(void *data, size_t length);
	void serialize(UnixPlusPlus::FileDesc &fd);

	uint32_t mechanism() 	{ return mMechanism; }
	uint64_t offset() 		{ return mOffset; }
	uint64_t length() 		{ return mLength; }

private:
	void clearData();

private:
	uint32_t mMechanism;
	uint64_t mOffset;
	uint64_t mLength;
	bool mFreeData;
	void *mData;
};

/// Represents the authentication table inside an encrypted disk image header.
class AuthTable {
public:
	AuthTable(UnixPlusPlus::FileDesc &fd);
	AuthTable() { }
	~AuthTable() { }

	std::vector<std::shared_ptr<AuthTableEntry>> &getEntries() { return mEntries; }
	void serialize(UnixPlusPlus::FileDesc &fd);
	void addEntry(uint32_t mechanism, void *data, size_t length);
	void prepareEntries();
	uint64_t findFirstEmptyDataOffset();

private:
	std::vector<std::shared_ptr<AuthTableEntry>> mEntries;
};

class EncDiskImageRep : public SingleDiskRep {
public:
	EncDiskImageRep(const char *path);
	virtual ~EncDiskImageRep();

	CFDataRef identification();
	CFDataRef component(CodeDirectory::SpecialSlot slot);
	size_t signingLimit();
	size_t signingBase();
	void strictValidate(const CodeDirectory* cd, const ToleratedErrors& tolerated, SecCSFlags flags);
	std::string format();
	void prepareForSigning(SigningContext& state);

	static bool candidate(UnixPlusPlus::FileDesc &fd);
	void registerStapledTicket();
	CFDataRef copyStapledTicket();

	static CFDataRef identificationFor(MachO *macho);
	void flush();

	DiskRep::Writer *writer();
	class Writer;
	friend class Writer;

private:
	void setup();
	static bool readHeader(UnixPlusPlus::FileDesc& fd, struct __Encrypted_Header_V2& header);

private:
	struct __Encrypted_Header_V2 mHeader;		// disk image header (all fields NBO)
	const EmbeddedSignatureBlob *mSigningData;	// pointer to signature SuperBlob (malloc'd memory during setup)
	AuthTable 					 mAuthTable;
};

class EncDiskImageRep::Writer : public SingleDiskRep::Writer, private EmbeddedSignatureBlob::Maker {
	friend class EncDiskImageRep;
public:
	Writer(EncDiskImageRep *r) : SingleDiskRep::Writer(r, writerNoGlobal), rep(r), mSigningData(NULL) { }
	void component(CodeDirectory::SpecialSlot slot, CFDataRef data);
	void flush();
	void remove();
	void addDiscretionary(CodeDirectory::Builder &builder) { }

private:
	EncDiskImageRep *rep;
	EmbeddedSignatureBlob *mSigningData;
};

} // end namespace CodeSigning
} // end namespace Security

#endif /* _H_ENCDISKIMAGEREP */
