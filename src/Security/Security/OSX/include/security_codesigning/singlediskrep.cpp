/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 17, 2024.
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
#include "singlediskrep.h"
#include "csutilities.h"
#include <security_utilities/cfutilities.h>
#include <sys/stat.h>

namespace Security {
namespace CodeSigning {

using namespace UnixPlusPlus;


//
// Construct a SingleDiskRep
//
SingleDiskRep::SingleDiskRep(const std::string &path)
	: mPath(path)
{
}


//
// The default binary identification of a SingleDiskRep is the (SHA-1) hash
// of the entire file itself.
//
CFDataRef SingleDiskRep::identification()
{
	SHA1 hash;
	this->fd().seek(0);
	hashFileData(this->fd(), &hash);
	SHA1::Digest digest;
	hash.finish(digest);
	return makeCFData(digest, sizeof(digest));
}


//
// Both the canonical and main executable path of a SingleDiskRep is, well, its path.
//
CFURLRef SingleDiskRep::copyCanonicalPath()
{
	return makeCFURL(mPath);
}

string SingleDiskRep::mainExecutablePath()
{
	return mPath;
}


//
// The default signing limit is the size of the file.
// This will do unless the signing data gets creatively stuck in there somewhere.
//
size_t SingleDiskRep::signingLimit()
{
	return fd().fileSize();
}

//
// No executable segment in non-machO files.
//
size_t SingleDiskRep::execSegLimit(const Architecture *)
{
	return 0;
}

//
// A lazily opened read-only file descriptor for the path.
//
FileDesc &SingleDiskRep::fd()
{
	if (!mFd)
		mFd.open(mPath, O_RDONLY);
	return mFd;
}

//
// Flush cached state
//
void SingleDiskRep::flush()
{
	mFd.close();
}

//Check the magic darwinup xattr
bool SingleDiskRep::appleInternalForcePlatform() const
{
	return mFd.hasExtendedAttribute("com.apple.root.installed");
}

//
// The recommended identifier of a SingleDiskRep is, absent any better clue,
// the basename of its path.
//
string SingleDiskRep::recommendedIdentifier(const SigningContext &)
{
	return canonicalIdentifier(mPath);
}
	
	
//
// Paranoid validation
//
void SingleDiskRep::strictValidate(const CodeDirectory* cd, const ToleratedErrors& tolerated, SecCSFlags flags)
{
	DiskRep::strictValidate(cd, tolerated, flags);

	if (flags & kSecCSStripDisallowedXattrs) {
		if (fd().hasExtendedAttribute(XATTR_RESOURCEFORK_NAME)) {
			fd().removeAttr(XATTR_RESOURCEFORK_NAME);
		}
		if (fd().hasExtendedAttribute(XATTR_FINDERINFO_NAME)) {
			fd().removeAttr(XATTR_FINDERINFO_NAME);
		}
	}

	if (flags & kSecCSRestrictSidebandData && tolerated.find(errSecCSInvalidAssociatedFileData) == tolerated.end()) {
		if (fd().hasExtendedAttribute(XATTR_RESOURCEFORK_NAME)) {
			CFStringRef message = CFStringCreateWithFormat(kCFAllocatorDefault, NULL, CFSTR("Disallowed xattr %s found on %s"), XATTR_RESOURCEFORK_NAME, mPath.c_str());
			CSError::throwMe(errSecCSInvalidAssociatedFileData, kSecCFErrorResourceSideband, message);
		}
		if (fd().hasExtendedAttribute(XATTR_FINDERINFO_NAME)) {
			CFStringRef message = CFStringCreateWithFormat(kCFAllocatorDefault, NULL, CFSTR("Disallowed xattr %s found on %s"), XATTR_FINDERINFO_NAME, mPath.c_str());
			CSError::throwMe(errSecCSInvalidAssociatedFileData, kSecCFErrorResourceSideband, message);
		}
	}

	// code limit must cover (exactly) the entire file
	if (cd && cd->signingLimit() != signingLimit())
		MacOSError::throwMe(errSecCSSignatureInvalid);
}



//
// Prototype Writers
//
FileDesc &SingleDiskRep::Writer::fd()
{
	if (!mFd)
		mFd.open(rep->path(), O_RDWR);
	return mFd;
}


} // end namespace CodeSigning
} // end namespace Security
