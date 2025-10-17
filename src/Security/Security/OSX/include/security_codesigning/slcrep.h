/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 16, 2025.
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
// slcrep - DiskRep representing the Mac OS Shared Library Cache
//
#ifndef _H_SLCREP
#define _H_SLCREP

#include "singlediskrep.h"
#include "sigblob.h"
#include <security_utilities/unix++.h>
#include <security_utilities/macho++.h>
#include <security_utilities/dyldcache.h>

namespace Security {
namespace CodeSigning {


//
// DYLDCacheRep implements the on-disk format for the Mac OS X
// Shared Library Cache, which coalesces a set of system libraries
// and frameworks into one big (mappable) code blob in the sky.
//
class DYLDCacheRep : public SingleDiskRep {
public:
	DYLDCacheRep(const Context *ctx = NULL);
	DYLDCacheRep(const char *path);
	
	CFDataRef component(CodeDirectory::SpecialSlot slot);
	size_t pageSize(const SigningContext &ctx);
	size_t signingLimit();
	std::string format();
	
	static bool candidate(UnixPlusPlus::FileDesc &fd);
	
public:
	static CFDataRef identificationFor(MachO *macho);
	
public:
	DiskRep::Writer *writer();
	class Writer;
	friend class Writer;

private:
	void setup();

private:
	DYLDCache mCache;
	const EmbeddedSignatureBlob *mSigningData;	// pointer to signature SuperBlob (in mapped memory)
};


//
// The write side of a FileDiskRep
//
class DYLDCacheRep::Writer : public SingleDiskRep::Writer, private EmbeddedSignatureBlob::Maker {
	friend class FileDiskRep;
public:
	Writer(DYLDCacheRep *r) : SingleDiskRep::Writer(r, writerNoGlobal), rep(r), mSigningData(NULL) { }
	void component(CodeDirectory::SpecialSlot slot, CFDataRef data);
	void flush();
	void addDiscretionary(CodeDirectory::Builder &builder);
	
private:
	DYLDCacheRep *rep;
	EmbeddedSignatureBlob *mSigningData;
};


} // end namespace CodeSigning
} // end namespace Security

#endif // !_H_SLCREP
