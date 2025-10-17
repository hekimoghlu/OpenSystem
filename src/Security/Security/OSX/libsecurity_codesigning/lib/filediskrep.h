/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 7, 2024.
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
// filediskrep - single-file executable disk representation
//
#ifndef _H_FILEDISKREP
#define _H_FILEDISKREP

#include "singlediskrep.h"
#include "machorep.h"
#include <security_utilities/cfutilities.h>

namespace Security {
namespace CodeSigning {


//
// A FileDiskRep represents a single code file on disk. We assume nothing about
// the format or contents of the file and impose no structure on it, other than
// assuming that all relevant code is contained in the file's data bytes.
// By default, we seal the entire file data as a single page.
//
// This is the ultimate fallback disk format. It is used if no other pattern
// applies. As such it is important that we do not introduce any assumptions
// here. Know that you do not know what any of the file means.
//
// FileDiskrep stores components in extended file attributes, one attribute
// per component. Note that this imposes size limitations on component size
// that may well be prohibitive in some applications.
//
// This DiskRep does not support resource sealing.
//
class FileDiskRep : public SingleDiskRep {
public:
	FileDiskRep(const char *path);
	
	CFDataRef component(CodeDirectory::SpecialSlot slot);
	std::string format();
	
	const Requirements *defaultRequirements(const Architecture *arch, const SigningContext &ctx);
	
public:
	DiskRep::Writer *writer();
	class Writer;
	friend class Writer;
	
protected:
	CFDataRef getAttribute(const char *name);
	static std::string attrName(const char *name);
};


//
// The write side of a FileDiskRep
//
class FileDiskRep::Writer : public SingleDiskRep::Writer {
	friend class FileDiskRep;
public:
	void component(CodeDirectory::SpecialSlot slot, CFDataRef data);
    void flush();
	void remove();
	bool preferredStore();

protected:
	Writer(FileDiskRep *r) : SingleDiskRep::Writer(r, writerLastResort) { }
	RefPointer<FileDiskRep> rep;
    std::set<std::string> mWrittenAttributes;
};


} // end namespace CodeSigning
} // end namespace Security

#endif // !_H_FILEDISKREP
