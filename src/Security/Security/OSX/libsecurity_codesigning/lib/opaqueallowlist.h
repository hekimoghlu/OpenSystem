/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 30, 2022.
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
#ifndef _H_OPAQUEALLOWLIST
#define _H_OPAQUEALLOWLIST

#include "SecAssessment.h"
#include <Security/CodeSigning.h>
#include <security_utilities/sqlite++.h>
#include <dispatch/dispatch.h>

namespace Security {
namespace CodeSigning {


namespace SQLite = SQLite3;


static const char opaqueDatabase[] = "/private/var/db/gkopaque.bundle/Contents/Resources/gkopaque.db";


class OpaqueAllowlist : public SQLite::Database {
public:
	OpaqueAllowlist(const char *path = NULL, int flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_NOFOLLOW);
	virtual ~OpaqueAllowlist();

public:
	void add(SecStaticCodeRef code);
	bool contains(SecStaticCodeRef code, SecAssessmentFeedback feedback, OSStatus reason);
	
	CFDictionaryRef validationConditionsFor(SecStaticCodeRef code);

private:
	dispatch_queue_t mOverrideQueue;
};


} // end namespace CodeSigning
} // end namespace Security

#endif //_H_OPAQUEALLOWLIST
