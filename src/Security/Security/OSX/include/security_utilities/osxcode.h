/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 12, 2022.
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
// osxcode - MacOS X's standard code objects
//
#ifndef _H_OSXCODE
#define _H_OSXCODE
#include <TargetConditionals.h>


#include <security_utilities/refcount.h>
#include <security_utilities/cfutilities.h>
#include <Security/CodeSigning.h>
#include <limits.h>
#include <string>
#include <vector>
#include <CoreFoundation/CFBundle.h>


namespace Security {


//
// A Signable with OS X support calls added
//
class OSXCode : public RefCount {
public:
	virtual ~OSXCode() { }
	
public:
	// creating OSXCode objects
	static RefPointer<OSXCode> main();
	static RefPointer<OSXCode> at(const char *path);
	static RefPointer<OSXCode> at(const std::string &path) { return at(path.c_str()); }
		
public:
	virtual string canonicalPath() const = 0;	// reverse of at()
	virtual string executablePath() const = 0;	// path to main executable file
	virtual SecStaticCodeRef codeRef() const;	// defaults to code at canonicalPath()

protected:
	OSXCode() { }	// nonpublic
};


//
// A simple executable tool.
//
class ExecutableTool : public OSXCode {
public:
	ExecutableTool(const char *path) : mPath(path) { }
	
	string path() const		{ return mPath; }
	string canonicalPath() const;
	string executablePath() const;
	
private:
	string mPath;			// UTF8 pathname to executable
};


//
// A generic bundle
//
class Bundle : public OSXCode {
public:
	Bundle(const char *path, const char *execPath = NULL);	// from root and optional exec path
	Bundle(CFBundleRef bundle, const char *root = NULL);		// from existing CFBundleRef
	~Bundle();

	string canonicalPath() const;
	string path() const				{ return mPath; }
	string executablePath() const;
	string identifier() const		{ return cfString(CFBundleGetIdentifier(cfBundle())); }
	CFTypeRef infoPlistItem(const char *name) const;	// not retained

	string resourcePath() const		{ return cfStringRelease(CFBundleCopyResourcesDirectoryURL(cfBundle())); }
	string resource(const char *name, const char *type, const char *subdir = NULL);
	void resources(vector<string> &paths, const char *type, const char *subdir = NULL);
	
	virtual void *lookupSymbol(const char *name);

protected:
	CFBundleRef cfBundle() const;
	
protected:
	string mPath;			// UTF8 path to bundle directory
	mutable string mExecutablePath;	// cached or determined path to main executable
	mutable CFBundleRef mBundle; // CF-style bundle object (lazily built)
};


class LoadableBundle : public Bundle {
public:
	LoadableBundle(const char *pathname) : Bundle(pathname) { }
	LoadableBundle(CFBundleRef bundle) : Bundle(bundle) { }
	
	virtual bool isLoaded() const;
	virtual void load();
	virtual void unload();
};


class OSXCodeWrap : public OSXCode {
public:
	OSXCodeWrap(SecStaticCodeRef code) : mCode(code) { }

	string encode() const;
	
	string canonicalPath() const;
	string executablePath() const;
	SecStaticCodeRef codeRef() const;

private:
	CFCopyRef<SecStaticCodeRef> mCode;
};


} // end namespace Security


#endif //_H_OSXCODE
