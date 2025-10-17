/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 5, 2025.
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
#include <sys/cdefs.h>
__FBSDID("$FreeBSD$");

#ifndef __APPLE__
#include <sys/capsicum.h>

#include <capsicum_helpers.h>
#endif
#include <err.h>
#include <errno.h>
#include <utmpx.h>

#include <algorithm>
#include <iostream>
#include <iterator>
#include <set>
#include <string>
using namespace std;

int
main(int argc, char **)
{
	struct utmpx *ut;
	set<string> names;

	if (argc > 1) {
		cerr << "usage: users" << endl;
		return (1);
	}

	setutxent();

#ifndef __APPLE__
	if (caph_enter())
		err(1, "Failed to enter capability mode.");
#endif
	while ((ut = getutxent()) != NULL)
		if (ut->ut_type == USER_PROCESS)
			names.insert(ut->ut_user);
	endutxent();

	if (!names.empty()) {
		set<string>::iterator last = names.end();
		--last;
		copy(names.begin(), last, ostream_iterator<string>(cout, " "));
		cout << *last << endl;
	}
}
