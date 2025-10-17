/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 7, 2022.
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
// quarantine++ - interface to XAR-format archive files
//
#ifndef _H_QUARANTINEPLUSPLUS
#define _H_QUARANTINEPLUSPLUS

#include <security_utilities/utilities.h>
#include <CoreFoundation/CoreFoundation.h>

extern "C" {
#include <quarantine.h>
}

namespace Security {
namespace CodeSigning {


//
// A file quarantine object
//
class FileQuarantine {
public:	
	FileQuarantine(const char *path);
	FileQuarantine(int fd);
	virtual ~FileQuarantine();
	
	uint32_t flags() const
		{ return qtn_file_get_flags(mQtn); }
	bool flag(uint32_t f) const
		{ return this->flags() & f; }
	
	void setFlags(uint32_t flags);
	void setFlag(uint32_t flag);
	void clearFlag(uint32_t flag);
	
	void applyTo(const char *path);
	void applyTo(int fd);
	
	operator bool() const { return mQtn != 0; }
	bool quarantined() const { return mQuarantined; }

private:
	void check(int err);
	
private:
	qtn_file_t mQtn;		// qtn handle
	bool mQuarantined;		// has quarantine information
};


} // end namespace CodeSigning
} // end namespace Security

#endif // !_H_QUARANTINEPLUSPLUS
