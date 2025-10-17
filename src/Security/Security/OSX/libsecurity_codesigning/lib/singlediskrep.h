/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 29, 2023.
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
// singlediskrep - semi-abstract diskrep for a single file of some kind
//
#ifndef _H_SINGLEDISKREP
#define _H_SINGLEDISKREP

#include "diskrep.h"
#include <security_utilities/unix++.h>

namespace Security {
namespace CodeSigning {


//
// A slight specialization of DiskRep that knows that it's working with a single
// file at a path that is both the canonical and main executable path. This is a common
// pattern.
//
// A SingleDiskRep is not a fully formed DiskRep in its own right. It must be further
// subclassed.
//
class SingleDiskRep : public DiskRep {
public:
	SingleDiskRep(const std::string &path);

	CFDataRef identification();								// partial file hash
	std::string mainExecutablePath();						// base path
	CFURLRef copyCanonicalPath();							// base path
	size_t signingLimit();									// size of file
	size_t execSegLimit(const Architecture *arch);			// size of executable segment
	UnixPlusPlus::FileDesc &fd();							// readable fd for this file
	void flush();											// close cached fd

	bool appleInternalForcePlatform() const;

	std::string recommendedIdentifier(const SigningContext &ctx); // basename(path)

	void strictValidate(const CodeDirectory* cd, const ToleratedErrors& tolerated, SecCSFlags flags);

public:
	class Writer;
	
protected:
	std::string path() const { return mPath; }

private:
	std::string mPath;
	UnixPlusPlus::AutoFileDesc mFd;							// open file (cached)
};


//
// A Writer for a SingleDiskRep
//
class SingleDiskRep::Writer : public DiskRep::Writer {
public:
	Writer(SingleDiskRep *r, uint32_t attrs = 0) : DiskRep::Writer(attrs), rep(r) { }

	UnixPlusPlus::FileDesc &fd();

private:
	RefPointer<SingleDiskRep> rep;							// underlying SingleDiskRep
	UnixPlusPlus::AutoFileDesc mFd;							// cached writable fd
};



} // end namespace CodeSigning
} // end namespace Security

#endif // !_H_SINGLEDISKREP
