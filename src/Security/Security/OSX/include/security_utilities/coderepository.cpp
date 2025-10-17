/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 7, 2023.
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
// CodeRepository - directory search paths for bundles
//
#include <security_utilities/coderepository.h>
#include <security_utilities/debugging.h>


namespace Security {


PathList::PathList()
{
}


PathList::PathList(const string &subPath,
		const char *suffix /* = NULL */,
		const char *envar /* = NULL */,
		bool forUser /* = true */)
	: mSuffix(suffix)
{
	if (envar)
		if (const char *envPath = getenv(envar)) {
#if !defined(NDEBUG)
			if (envPath[0] == '!') {
				// envar="!path" -> single-item override (debugging only)
				mDebugOverride = envPath + 1;
				secinfo("pathlist", "%p env(\"%s\") overrides to \"%s\"",
					this, envar, mDebugOverride.c_str());
				return;
			}
#endif //NDEBUG

			// treat envPath as a classic colon-separated list of directories
			secinfo("pathlist", "%p configuring from env(\"%s\")", this, envar);
			while (const char *p = strchr(envPath, ':')) {
				addDirectory(string(envPath, p - envPath));
				envPath = p + 1;
			}
			addDirectory(envPath);
			return;
		}

	// no joy from environment variables
	secinfo("pathlist", "%p configuring from default path set \"%s\"", this, subPath.c_str());
	if (forUser)
		secinfo("pathlist", "user search list not yet implemented");
	addDirectory("/Library/" + subPath);
	addDirectory("/System/Library/" + subPath);
}


PathList::~PathList()
{
	// virtual
}


void PathList::addDirectory(const string &dirPath)
{
	//@@@ validate dirPath?
	mPaths.push_back(dirPath);
}


}	// end namespace Security
