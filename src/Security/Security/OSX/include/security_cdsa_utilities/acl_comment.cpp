/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 3, 2023.
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
// acl_comment - "ignore" ACL subject type.
//
// CommentAclSubjects were a bad idea, badly implemented. The code below
// exists solely to keep existing (external) ACL forms from blowing up the
// ACL reader machinery and crashing the evaluation host.
// The original serialization code was not architecture independent - for either
// pointer sizes(!) or byte ordering. Yes, that was a stupid mistake.
// The following code is intentionally, wilfully violating the layer separation
// of the ACL reader/writer machine to deduce enough information about the
// originating architecture to cleanly consume (just) the bytes making up this
// ACL's external representation. We make no use of the bytes read; thankfully,
// the semantics of a CommentAclSubject have always been "never matches."
// We do not preserve them on write-out; a newly-written ACL will contain no data
// (and will read cleanly).
// If you use this code as a template for anything (other than a how-not-to-write-code
// seminar), your backups shall rot right after your main harddrive crashes, and
// you have only yourself to blame.
//
#include <security_cdsa_utilities/acl_comment.h>
#include <security_cdsa_utilities/cssmwalkers.h>
#include <security_cdsa_utilities/cssmlist.h>
#include <algorithm>

using namespace DataWalkers;


//
// The COMMENT subject matches nothing, no matter how pretty.
//
bool CommentAclSubject::validates(const AclValidationContext &) const
{
	return false;
}


//
// The list form has no values.
//
CssmList CommentAclSubject::toList(Allocator &alloc) const
{
	return TypedList(Allocator::standard(), CSSM_ACL_SUBJECT_TYPE_COMMENT);
}


//
// We completely disregard any data contained in CSSM form COMMENT ACLs.
//
CommentAclSubject *CommentAclSubject::Maker::make(const TypedList &list) const
{
	return new CommentAclSubject();
}


//
// This is the nasty code. We don't really care what data was originally baked
// into this ACL's external (stream) form, but since there's no external framing
// to delimit it, we need to figure out how many bytes to consume to keep the
// reader from going out of sync. And that's not pretty, since the external form
// contains (stupidly!) a pointer, so we have all permutations of byte order and
// pointer size to worry about.
//
CommentAclSubject *CommentAclSubject::Maker::make(Version, Reader &pub, Reader &) const
{
	//
	// At this point, the Reader is positioned at data that was once written using
	// this code:
	//	pub(ptr);  // yes, that's a pointer
    //	pub.countedData(ptr, size);
	// We know ptr was a non-NULL pointer (4 or 8 bytes, alas).
	// CountedData writes a 4-byte NBO length followed by that many bytes.
	// The data written starts with a CSSM_LIST structure in native architecture.
	// That in turn begins with a CSSM_LIST_TYPE (4 bytes, native, 0<=type<=2).
	// So to summarize (h=host byte order, n=network byte order), we might be looking at:
	//   32 bits:  | P4h | L4n | T4h | (L-4 bytes) |
	//   64 bits:  |    P8h    | L4n |  (L bytes)  |
	// It's the T4h-or-L4n bytes that save our day, since we know that
	//	0 <= T <= 2 (definition of CSSM_LIST_TYPE)
	//	16M > L >= sizeof(CSSM_LIST) >= 12
	// Phew. I'd rather be lucky than good...
	//
	// So let's get started:
	static const size_t minCssmList = 12;	// min(sizeof(CSSM_LIST)) of all architectures
	pub.get<void>(4);			// skip first 4 bytes
	uint32_t lop; pub(lop);		// read L4n-or-(bottom of)P8h
	uint32_t tol; pub(tol);		// read T4h-or-L4n
	if (tol <= 2 || flip(tol) <= 2) {	// 32 bits
		// the latter can't be a very big (flipped) L because we know 12 < L < 16M,
		// and you'd have to be a multiple of 2^24 to pass that test
		size_t length = n2h(lop);
		assert(length >= minCssmList);
		pub.get<void>(length - sizeof(tol)); // skip L-4 bytes
	} else {							// 64 bits
		size_t length = n2h(tol);
		assert(length >= minCssmList);
		pub.get<void>(length); // skip L bytes
	}
	
	// we've successfully thrown out the garbage. What's left is a data-less subject
	return new CommentAclSubject();		// no data
}


//
// Export to blob form.
// This simply writes the smallest form consistent with the heuristic above.
//
void CommentAclSubject::exportBlob(Writer::Counter &pub, Writer::Counter &)
{
	uint32_t zero = 0;
	Endian<uint32_t> length = 12;
	pub(zero); pub(length); pub(zero); pub(zero); pub(zero);
}

void CommentAclSubject::exportBlob(Writer &pub, Writer &)
{
	uint32_t zero = 0;
	Endian<uint32_t> length = 12;
	pub(zero); pub(length); pub(zero); pub(zero); pub(zero);
}


#ifdef DEBUGDUMP

void CommentAclSubject::debugDump() const
{
	Debug::dump("Comment[never]");
}

#endif //DEBUGDUMP

CFStringRef CommentAclSubject::createACLDebugString() const
{
    return CFStringCreateWithFormat(kCFAllocatorDefault, NULL, CFSTR("<CommentAclSubject>"));
}
