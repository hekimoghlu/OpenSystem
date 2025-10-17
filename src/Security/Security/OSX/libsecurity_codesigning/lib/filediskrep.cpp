/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 1, 2024.
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
#include "filediskrep.h"
#include "StaticCode.h"
#include <security_utilities/macho++.h>
#include <cstring>


namespace Security {
namespace CodeSigning {

using namespace UnixPlusPlus;


//
// Everything's lazy in here
//
FileDiskRep::FileDiskRep(const char *path)
	: SingleDiskRep(path)
{
	CODESIGN_DISKREP_CREATE_FILE(this, (char*)path);
}


//
// Produce an extended attribute name from a canonical slot name
//
string FileDiskRep::attrName(const char *name)
{
	static const char prefix[] = "com.apple.cs.";
	return string(prefix) + name;
}


//
// Retrieve an extended attribute by name
//
CFDataRef FileDiskRep::getAttribute(const char *name)
{
	string aname = attrName(name);
	try {
		ssize_t length = fd().getAttrLength(aname);
		if (length < 0)
			return NULL;		// no such attribute
		CFMallocData buffer(length);
		fd().getAttr(aname, buffer, length);
		return buffer;
	} catch (const UnixError &err) {
		// recover some errors that happen in (relatively) benign circumstances
		switch (err.error) {
		case ENOTSUP:	// no extended attributes on this filesystem
		case EPERM:		// filesystem objects to name(?)
			return NULL;
		default:
			throw;
		}
	}
}


//
// Extract and return a component by slot number.
// If we have a Mach-O binary, use embedded components.
// Otherwise, look for and return the extended attribute, if any.
//
CFDataRef FileDiskRep::component(CodeDirectory::SpecialSlot slot)
{
	if (const char *name = CodeDirectory::canonicalSlotName(slot))
		return getAttribute(name);
	else
		return NULL;
}


//
// Generate a suggested set of internal requirements.
// We don't really have to say much. However, if we encounter a file that
// starts with the magic "#!" script marker, we do suggest that this should
// be a valid host if we can reasonably make out what that is.
//
const Requirements *FileDiskRep::defaultRequirements(const Architecture *, const SigningContext &ctx)
{
	// read start of file
	char buffer[256];
	size_t length = fd().read(buffer, sizeof(buffer), 0);
	if (length > 3 && buffer[0] == '#' && buffer[1] == '!' && buffer[2] == '/') {
		// isolate (full) path element in #!/full/path -some -other -stuff
		if (length == sizeof(buffer))
			length--;
		buffer[length] = '\0';
		char *cmd = buffer + 2;
		cmd[strcspn(cmd, " \t\n\r\f")] = '\0';
		secinfo("filediskrep", "looks like a script for %s", cmd);
		if (cmd[1])
			try {
				// find path on disk, get designated requirement (if signed)
				string path = ctx.sdkPath(cmd);
				if (RefPointer<DiskRep> rep = DiskRep::bestFileGuess(path))
					if (SecPointer<SecStaticCode> code = new SecStaticCode(rep))
						if (const Requirement *req = code->designatedRequirement()) {
							CODESIGN_SIGN_DEP_INTERP(this, (char*)cmd, (void*)req);
							// package up as host requirement and return that
							Requirements::Maker maker;
							maker.add(kSecHostRequirementType, req->clone());
							return maker.make();
						}
			} catch (...) {
				secinfo("filediskrep", "exception getting host requirement (ignored)");
			}
	}
	return NULL;
}


string FileDiskRep::format()
{
	return "generic";
}

//
// FileDiskRep::Writers
//
DiskRep::Writer *FileDiskRep::writer()
{
	return new Writer(this);
}


//
// Write a component.
// Note that this isn't concerned with Mach-O writing; this is handled at
// a much higher level. If we're called, it's extended attribute time.
//
void FileDiskRep::Writer::component(CodeDirectory::SpecialSlot slot, CFDataRef data)
{
	try {
        std::string name = attrName(CodeDirectory::canonicalSlotName(slot));
		fd().setAttr(name, CFDataGetBytePtr(data), CFDataGetLength(data));
        mWrittenAttributes.insert(name);
	} catch (const UnixError &error) {
		if (error.error == ERANGE)
			MacOSError::throwMe(errSecCSCMSTooLarge);
		throw;
	}
}
    

void FileDiskRep::Writer::flush()
{
    size_t size = fd().listAttr(NULL, 0);
    std::vector<char> buffer(size);
    char *s = &buffer[0];
    char *end = &buffer[size];
    fd().listAttr(s, size);
    while (s < end) {
        std::string name = s;
        s += strlen(s) + 1;     // skip to next
        if (name.compare(0, 13, "com.apple.cs.") == 0)  // one of ours
            if (mWrittenAttributes.find(name) == mWrittenAttributes.end()) {    // not written by this signing operation
                fd().removeAttr(name);
        }
    }
}


//
// Clear all signing data
//
void FileDiskRep::Writer::remove()
{
	for (CodeDirectory::SpecialSlot slot = 0; slot < cdSlotCount; slot++)
		if (const char *name = CodeDirectory::canonicalSlotName(slot))
			fd().removeAttr(attrName(name));
	fd().removeAttr(attrName(kSecCS_SIGNATUREFILE));
}


//
// We are NOT the preferred store for components because our approach
// (extended attributes) suffers from some serious limitations.
//
bool FileDiskRep::Writer::preferredStore()
{
	return false;
}


} // end namespace CodeSigning
} // end namespace Security
