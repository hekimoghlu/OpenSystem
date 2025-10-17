/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 29, 2022.
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
// dyldcache - access layer to the DYLD Shared Library Cache file
//
#include "dyldcache.h"


//
// Table of supported architectures.
// The cache file header has no direct architecture information, so we need to deduce it like this.
//
static const uint16_t bigEndian = 0x1200;
static const uint16_t littleEndian = 0x0012;

const DYLDCache::ArchType DYLDCache::architectures[] = {
	{ CPU_TYPE_X86_64, CPU_SUBTYPE_MULTIPLE,	"dyld_v1  x86_64", "x86_64", littleEndian },
	{ CPU_TYPE_X86, CPU_SUBTYPE_MULTIPLE,		"dyld_v1    i386", "i386", littleEndian },
	{ CPU_TYPE_POWERPC, CPU_SUBTYPE_MULTIPLE,	"dyld_v1     ppc", "rosetta", bigEndian },
	{ CPU_TYPE_ARM, CPU_SUBTYPE_ARM_V6,			"dyld_v1   armv6", "armv6", littleEndian },
	{ CPU_TYPE_ARM, CPU_SUBTYPE_ARM_V7,			"dyld_v1   armv7", "armv7", littleEndian },
	{ 0 }
};

const DYLDCache::ArchType DYLDCache::defaultArchitecture =
	{ 0, 0, "dyld_v1 default", "unknown", littleEndian };


//
// Architecture matching and lookup
//
std::string DYLDCache::pathFor(const Architecture &arch)
{
	for (const ArchType *it = architectures; it->cpu; it++)
		if (arch.matches(it->architecture()))
			return it->path();
	UnixError::throwMe(ENOEXEC);
}

const DYLDCache::ArchType *DYLDCache::matchArchitecture(const dyld_cache_header &header)
{
	for (const ArchType *arch = architectures; arch->cpu; arch++)
		if (!strcmp(header.magic, arch->magic))
			return arch;
	if (!strncmp(header.magic, "dyld_v1 ", 8))
		return &defaultArchitecture;
	return NULL;
}


//
// Construction and teardown
//
DYLDCache::DYLDCache(const std::string &path)
{
	this->open(path);
	mLength = this->fileSize();
	mBase = this->mmap(PROT_READ, mLength);
	mHeader = at<dyld_cache_header>(0);

	if ((mArch = matchArchitecture(*mHeader)) == NULL)
		UnixError::throwMe(ENOEXEC);
	mFlip = *((const uint8_t *)&mArch->order) != 0x12;
	
	mSigStart = (size_t)flip(mHeader->codeSignatureOffset);
	mSigLength = (size_t)flip(mHeader->codeSignatureSize);
	size_t sigEnd = mSigStart + mSigLength;
	if (mSigStart > sigEnd || sigEnd > mLength)
		UnixError::throwMe(ENOEXEC);
}


DYLDCache::~DYLDCache()
{
	::munmap((void *)mBase, mLength);
}


//
// Preflight a file for file type
//
bool DYLDCache::validate(UnixPlusPlus::FileDesc &fd)
{
	dyld_cache_header header;
	return fd.read(&header, sizeof(header), 0) == sizeof(header)
		&& matchArchitecture(header) != NULL;
}


//
// Locate a mapping in the cache
//
DYLDCache::Mapping DYLDCache::mapping(unsigned ix) const
{
	assert(ix < this->mappingCount());
	return Mapping(*this, flip(mHeader->mappingOffset) + ix * sizeof(dyld_cache_mapping_info));
}


//
// Locate an image in the cache
//
DYLDCache::Image DYLDCache::image(unsigned ix) const
{
	assert(ix < this->imageCount());
	return Image(*this, flip(mHeader->imagesOffset) + ix * sizeof(dyld_cache_image_info));
}



DYLDCache::Mapping DYLDCache::findMap(uint64_t address) const
{
	for (unsigned ix = 0; ix < mappingCount(); ix++) {
		Mapping map = this->mapping(ix);
		if (map.contains(address))
			return map;
	}
	UnixError::throwMe(EINVAL);
}

uint64_t DYLDCache::mapAddress(uint64_t address) const
{
	Mapping map = this->findMap(address);
	return (address - map.address()) + map.offset();
}
