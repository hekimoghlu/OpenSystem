/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 9, 2025.
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
#ifndef _H_DIRSCANNER
#define _H_DIRSCANNER

#include "resources.h"
#include <dirent.h>
#include <fts.h>
#include <security_utilities/cfutilities.h>

namespace Security {
namespace CodeSigning {


class DirScanner {
public:
	DirScanner(const char *path);
	DirScanner(string path);
	~DirScanner();

	struct dirent *getNext();	// gets the next item out of this DirScanner
	bool initialized();			// returns false if the constructor failed to initialize the dirent
	
	void unlink(const struct dirent* ent, int flags);
	bool isRegularFile(dirent* dp);

private:
	string path;
	DIR *dp = NULL;
	struct dirent entBuffer;
	void initialize();
	bool init;
};


class DirValidator {
public:
	DirValidator() : mRequireCount(0) { }
	~DirValidator();

	enum {
		file = 0x01,
		directory = 0x02,
		symlink = 0x04,
		noexec = 0x08,
		required = 0x10,
		descend = 0x20,
	};

	typedef std::string (^TargetPatternBuilder)(const std::string &name, const std::string &target);

private:
	class Rule : public ResourceBuilder::Rule {
	public:
		Rule(const std::string &pattern, uint32_t flags, TargetPatternBuilder targetBlock);
		~Rule();

		bool matchTarget(const char *path, const char *target) const;

	private:
		TargetPatternBuilder mTargetBlock;
	};
	void addRule(Rule *rule) { mRules.push_back(rule); }

	class FTS {
	public:
		FTS(const std::string &path, int options = FTS_PHYSICAL | FTS_COMFOLLOW | FTS_NOCHDIR);
		~FTS();

		operator ::FTS* () const { return mFTS; }

	private:
		::FTS *mFTS;
	};

public:
	void allow(const std::string &namePattern, uint32_t flags, TargetPatternBuilder targetBlock = NULL)
	{ addRule(new Rule(namePattern, flags, targetBlock)); }
	void require(const std::string &namePattern, uint32_t flags, TargetPatternBuilder targetBlock = NULL)
	{ addRule(new Rule(namePattern, flags | required, targetBlock)); mRequireCount++; }

	void allow(const std::string &namePattern, uint32_t flags, std::string targetPattern)
	{ allow(namePattern, flags, ^ string (const std::string &name, const std::string &target) { return targetPattern; }); }
	void require(const std::string &namePattern, uint32_t flags, std::string targetPattern)
	{ require(namePattern, flags, ^ string (const std::string &name, const std::string &target) { return targetPattern; }); }

	void validate(const std::string &root, OSStatus error);

private:
	Rule * match(const char *relpath, uint32_t flags, bool executable, const char *target = NULL);

private:
	typedef std::vector<Rule *> Rules;
	Rules mRules;
	int mRequireCount;
};


} // end namespace CodeSigning
} // end namespace Security

#endif // !_H_DIRSCANNER
