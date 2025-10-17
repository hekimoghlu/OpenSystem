/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 8, 2023.
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
// reqreader - Requirement language (exprOp) reader/scanner
//
#include "reqreader.h"
#include <Security/SecTrustSettingsPriv.h>
#include <security_utilities/memutils.h>

#if TARGET_OS_OSX
#include <security_cdsa_utilities/cssmdata.h>	// for hex encoding
#endif

#include "csutilities.h"

namespace Security {
namespace CodeSigning {


//
// Requirement::Reader
//
Requirement::Reader::Reader(const Requirement *req)
	: mReq(req), mPC(sizeof(Requirement))
{
	assert(req);
	if (req->kind() != exprForm && req->kind() != lwcrForm)
		MacOSError::throwMe(errSecCSReqUnsupported);
}


//
// Access helpers to retrieve various data types from the data stream
//
void Requirement::Reader::getData(const void *&data, size_t &length)
{
	length = get<uint32_t>();
	checkSize(length);
	data = (mReq->at<void>(mPC));
	mPC += LowLevelMemoryUtilities::alignUp(length, baseAlignment);
}

string Requirement::Reader::getString()
{
	const char *s; size_t length;
	getData(s, length);
	return string(s, length);
}

CFDataRef Requirement::Reader::getHash()
{
	const unsigned char *s; size_t length;
	getData(s, length);
	return makeCFData(s, length);
}

CFAbsoluteTime Requirement::Reader::getAbsoluteTime()
{
	// timestamps are saved as 64bit ints internally for
	// portability, but CoreFoundation wants CFAbsoluteTimes,
	// which are doubles.
	// This cuts off subseconds.
	return static_cast<CFAbsoluteTime>(get<int64_t>());
}

const unsigned char *Requirement::Reader::getSHA1()
{
	const unsigned char *digest; size_t length;
	getData(digest, length);
	if (length != CC_SHA1_DIGEST_LENGTH)
		MacOSError::throwMe(errSecCSReqInvalid);
	return digest;
}

void Requirement::Reader::skip(size_t length)
{
	checkSize(length);
	mPC += length;
}


}	// CodeSigning
}	// Security
