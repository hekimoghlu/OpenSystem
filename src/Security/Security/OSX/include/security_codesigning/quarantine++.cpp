/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 18, 2024.
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
// xar++ - interface to XAR-format archive files
//
#include "quarantine++.h"


namespace Security {
namespace CodeSigning {


//
// Check the int result of a qtn API call.
// If the error is "not quarantined," note in the object (no error).
// Other qtn-specific errors are arbitrarily mapped to ENOSYS (this isn't
// important enough to subclass CommonError).
//
void FileQuarantine::check(int err)
{
	switch (err) {
	case 0:
		mQuarantined = true;
		break;
	case QTN_NOT_QUARANTINED:
		mQuarantined = false;
		return;
	default:	// some flavor of quarantine-not-available
		UnixError::throwMe(err);
	}
}


FileQuarantine::~FileQuarantine()
{
	if (mQtn)
		qtn_file_free(mQtn);
}


FileQuarantine::FileQuarantine(const char *path)
{
	if (!(mQtn = qtn_file_alloc()))
		UnixError::throwMe();
	check(qtn_file_init_with_path(mQtn, path));
}

FileQuarantine::FileQuarantine(int fd)
{
	if (!(mQtn = qtn_file_alloc()))
		UnixError::throwMe();
	check(qtn_file_init_with_fd(mQtn, fd));
}


void FileQuarantine::setFlags(uint32_t flags)
{
	if (mQuarantined)
		check(qtn_file_set_flags(mQtn, flags));
}

void FileQuarantine::setFlag(uint32_t flag)
{
	if (mQuarantined)
		setFlags(flags() | flag);
}

void FileQuarantine::clearFlag(uint32_t flag)
{
	if (mQuarantined)
		setFlags(flags() & ~flag);
}

void FileQuarantine::applyTo(const char *path)
{
	check(qtn_file_apply_to_path(mQtn, path));
}

void FileQuarantine::applyTo(int fd)
{
	check(qtn_file_apply_to_fd(mQtn, fd));
}


} // end namespace CodeSigning
} // end namespace Security
