/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 18, 2022.
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
// bundlerepository - directory search paths for bundles
//
#ifndef _H_BUNDLEREPOSITORY
#define _H_BUNDLEREPOSITORY

#include <security_utilities/cfutilities.h>
#include <security_utilities/refcount.h>
#include <CoreFoundation/CoreFoundation.h>
#include <string>
#include <vector>


namespace Security {


//
// PathList abstracts a directory search path.
// It's not really powerful enough to be useful on its own.
//
class PathList {
public:
	PathList();
	PathList(const string &subPath, const char *suffix = NULL,
		const char *envar = NULL, bool forUser = true);
	virtual ~PathList();
	
	void addDirectory(const string &dirPath);

protected:
	vector<string> mPaths;
	string mSuffix;
	IFDEBUG(string mDebugOverride);
};


//
// CodeRepository<Code> represents all code objects within the PathList search path,
// represented forcibly as objects of type Code.
//
template <class Code>
class CodeRepository : public vector<RefPointer<Code> >, public PathList {
public:
	CodeRepository() { }				// empty - populate with paths
	CodeRepository(const string &subPath, const char *suffix = NULL,
		const char *envar = NULL, bool forUser = true)
		: PathList(subPath, suffix, envar, forUser) { }
	
	void update();
};


//
// The generic implementation of update works with subclasses of GenericBundle,
// represented through CFBundleRefs collected via CFBundle.
// (Technically, this would work with anything that has a constructor from CFBundleRef.)
// If we ever wanted a CodeRepository<ExecutableTool>, we'd specialize update() to deal with
// ExecutableTool's slightly different constructor.
//
template <class Code>
void CodeRepository<Code>::update()
{
	vector<RefPointer<Code> > result;
	for (vector<string>::const_iterator it = mPaths.begin(); it != mPaths.end(); it++) {
		if (CFRef<CFArrayRef> bundles = CFBundleCreateBundlesFromDirectory(NULL,
				CFTempURL(*it, true), mSuffix.empty() ? NULL : CFStringRef(CFTempString(mSuffix)))) {
			CFIndex count = CFArrayGetCount(bundles);
			secinfo("coderep", "%p directory %s has %ld entries", this, it->c_str(), count);
			for (CFIndex n = 0; n < count; n++)
				try {
					result.push_back(new Code((CFBundleRef)CFArrayGetValueAtIndex(bundles, n)));
				} catch (...) {
					secinfo("coderep", "%p exception creating %s (skipped)",
						this, cfString(CFBundleRef(CFArrayGetValueAtIndex(bundles, n))).c_str());
				}
		} else
			secinfo("coderep", "directory %s bundle read failed", it->c_str());
	}
	secinfo("coderep", "%p total of %ld items in list", this, result.size());
	this->swap(result);
}


} // end namespace Security

#endif //_H_BUNDLEREPOSITORY
