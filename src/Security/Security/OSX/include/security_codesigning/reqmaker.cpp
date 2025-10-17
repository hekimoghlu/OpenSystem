/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 19, 2024.
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
// reqmaker - Requirement assembler
//
#include "reqmaker.h"

namespace Security {
namespace CodeSigning {


//
// Requirement::Makers
//
Requirement::Maker::Maker(Kind k)
	: mSize(1024)
{
	mBuffer = (Requirement *)malloc(mSize);
	mBuffer->initialize();
	mBuffer->kind(k);
	mPC = sizeof(Requirement);
}

// need at least (size) bytes in the creation buffer
void Requirement::Maker::require(size_t size)
{
	if (mPC + size > mSize) {
		mSize *= 2;
		if (mPC + size > mSize)
			mSize = (Offset)(mPC + size);
		if (!(mBuffer = (Requirement *)realloc(mBuffer, mSize)))
			UnixError::throwMe(ENOMEM);
	}
}

// allocate (size) bytes at end of buffer and return pointer to that
void *Requirement::Maker::alloc(size_t size)
{
	// round size up to preserve alignment
	size_t usedSize = LowLevelMemoryUtilities::alignUp(size, baseAlignment);
	require(usedSize);
	void *data = mBuffer->at<void>(mPC);
	mPC += usedSize;
	
	// clear any padding (avoid random bytes in code image)
	const uint32_t zero = 0;
	memcpy(mBuffer->at<void>(mPC - usedSize + size), &zero, usedSize - size);
	
	// all done
	return data;
}

// put contiguous data blob
void Requirement::Maker::putData(const void *data, size_t length)
{
	put(uint32_t(length));
	memcpy(alloc(length), data, length);
}

// Specialized Maker put operations
void Requirement::Maker::anchor()
{
	put(opAppleAnchor);
}

void Requirement::Maker::anchorGeneric()
{
	put(opAppleGenericAnchor);
}

void Requirement::Maker::anchor(int slot, SHA1::Digest digest)
{
	put(opAnchorHash);
	put(slot);
	putData(digest, SHA1::digestLength);
}

void Requirement::Maker::anchor(int slot, const void *cert, size_t length)
{
	SHA1 hasher;
	hasher(cert, length);
	SHA1::Digest digest;
	hasher.finish(digest);
	anchor(slot, digest);
}

void Requirement::Maker::trustedAnchor()
{
	put(opTrustedCerts);
}

void Requirement::Maker::trustedAnchor(int slot)
{
	put(opTrustedCert);
	put(slot);
}

void Requirement::Maker::infoKey(const string &key, const string &value)
{
	put(opInfoKeyValue);
	put(key);
	put(value);
}

void Requirement::Maker::ident(const string &identifier)
{
	put(opIdent);
	put(identifier);
}

void Requirement::Maker::cdhash(SHA1::Digest digest)
{
	put(opCDHash);
	putData(digest, SHA1::digestLength);
}

void Requirement::Maker::cdhash(CFDataRef digest)
{
	put(opCDHash);
	putData(CFDataGetBytePtr(digest), CFDataGetLength(digest));
}
	
void Requirement::Maker::platform(int platformIdentifier)
{
	put(opPlatform);
	put(platformIdentifier);
}


void Requirement::Maker::copy(const Requirement *req)
{
	assert(req);
	if (req->kind() != exprForm)		// don't know how to embed this
		MacOSError::throwMe(errSecCSReqUnsupported);
	this->copy(req->at<const void>(sizeof(Requirement)), req->length() - sizeof(Requirement));
}


void *Requirement::Maker::insert(const Label &label, size_t length)
{
	require(length);
	memmove(mBuffer->at<void>(label.pos + length),
		mBuffer->at<void>(label.pos), mPC - label.pos);
	mPC += length;
	return mBuffer->at<void>(label.pos);
}


Requirement *Requirement::Maker::make()
{
	mBuffer->length(mPC);
	Requirement *result = mBuffer;
	mBuffer = NULL;
	return result;
}


}	// CodeSigning
}	// Security
