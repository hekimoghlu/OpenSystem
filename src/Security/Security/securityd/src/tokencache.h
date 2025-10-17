/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 28, 2024.
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
// tokencache - persistent (on-disk) hardware token directory
//
#ifndef _H_TOKENCACHE
#define _H_TOKENCACHE

#include <security_utilities/refcount.h>
#include <Security/cssm.h>


//
// A little helper
//
class Rooted {
public:
	Rooted() { }
	Rooted(const char *root) : mRoot(root) { }
	Rooted(const string &root) : mRoot(root) { }
	
	string root() const { return mRoot; }
	string path(const char *sub) const;
	string path(const string &sub) const { return path(sub.c_str()); }

protected:
	void root(const string &s);

private:
	string mRoot;				// root of this tree
};


//
// An on-disk cache area.
// You'll only want a single one, though nothing keeps you from
// making multiples if you like.
//
class TokenCache : public Rooted {
public:
	TokenCache(const char *root);
	~TokenCache();
	
	uid_t tokendUid() const { return mTokendUid; }
	gid_t tokendGid() const { return mTokendGid; }
	
public:
	class Token : public RefCount, public Rooted {
	public:
		friend class TokenCache;
		Token(TokenCache &cache, const std::string &uid);
		Token(TokenCache &cache);
		~Token();
		
		enum Type { existing, created, temporary };
		Type type() const { return mType; }

		TokenCache &cache;
		uint32 subservice() const { return mSubservice; }
		string workPath() const;
		string cachePath() const;
		
		string printName() const;
		void printName(const string &name);
		
		uid_t tokendUid() const { return cache.tokendUid(); }
		gid_t tokendGid() const { return cache.tokendGid(); }
	
	protected:		
		void init(Type type);

	private:
		uint32 mSubservice;		// subservice id assigned
		Type mType;				// type of Token cache entry
	};

public:
	uint32 allocateSubservice();

private:
	enum Owner { securityd, tokend };
	void makedir(const char *path, int flags, mode_t mode, Owner owner);
	void makedir(const string &path, int flags, mode_t mode, Owner owner)
	{ return makedir(path.c_str(), flags, mode, owner); }
	
private:
	uint32 mLastSubservice; // last subservice id issued

	uid_t mTokendUid;		// uid of daemons accessing this token cache
	gid_t mTokendGid;		// gid of daemons accessing this token cache
};


#endif //_H_TOKENCACHE
