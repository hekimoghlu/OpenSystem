/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 14, 2025.
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
#ifndef _H_POLICYENGINE
#define _H_POLICYENGINE

#include "SecAssessment.h"
#include "opaqueallowlist.h"
#include "evaluationmanager.h"
#include "policydb.h"
#include <security_utilities/globalizer.h>
#include <security_utilities/cfutilities.h>
#include <security_utilities/hashing.h>
#include <security_utilities/sqlite++.h>
#include <CoreFoundation/CoreFoundation.h>
#include <Security/CodeSigning.h>

namespace Security {
namespace CodeSigning {


typedef uint EngineOperation;
enum {
	opInvalid = 0,
	opEvaluate,
	opAddAuthority,
	opRemoveAuthority,
};


class PolicyEngine : public PolicyDatabase {
public:
	PolicyEngine();
	PolicyEngine(const char *path);
	virtual ~PolicyEngine();

public:
	void evaluate(CFURLRef path, AuthorityType type, SecAssessmentFlags flags, CFDictionaryRef context, CFMutableDictionaryRef result);

	CFDictionaryRef update(CFTypeRef target, SecAssessmentFlags flags, CFDictionaryRef context);
	CFDictionaryRef add(CFTypeRef target, AuthorityType type, SecAssessmentFlags flags, CFDictionaryRef context);
	CFDictionaryRef remove(CFTypeRef target, AuthorityType type, SecAssessmentFlags flags, CFDictionaryRef context);
	CFDictionaryRef enable(CFTypeRef target, AuthorityType type, SecAssessmentFlags flags, CFDictionaryRef context, bool authorize);
	CFDictionaryRef disable(CFTypeRef target, AuthorityType type, SecAssessmentFlags flags, CFDictionaryRef context, bool authorize);
	CFDictionaryRef find(CFTypeRef target, AuthorityType type, SecAssessmentFlags flags, CFDictionaryRef context);

	void recordFailure(CFDictionaryRef info);

public:
	static void addAuthority(SecAssessmentFlags flags, CFMutableDictionaryRef parent, const char *label, SQLite::int64 row = 0, CFTypeRef cacheInfo = NULL, bool weak = false, uint64_t ruleFlags = 0);
	static void addToAuthority(CFMutableDictionaryRef parent, CFStringRef key, CFTypeRef value);

private:
	void evaluateCode(CFURLRef path, AuthorityType type, SecAssessmentFlags flags, CFDictionaryRef context, CFMutableDictionaryRef result, bool handleUnsigned);
	void evaluateInstall(CFURLRef path, SecAssessmentFlags flags, CFDictionaryRef context, CFMutableDictionaryRef result);
	void evaluateDocOpen(CFURLRef path, SecAssessmentFlags flags, CFDictionaryRef context, CFMutableDictionaryRef result);
	
	void evaluateCodeItem(SecStaticCodeRef code, CFURLRef path, AuthorityType type, SecAssessmentFlags flags, bool nested, CFMutableDictionaryRef result);
	void adjustValidation(SecStaticCodeRef code);
	bool temporarySigning(SecStaticCodeRef code, AuthorityType type, CFURLRef path, SecAssessmentFlags matchFlags);
	void normalizeTarget(CFRef<CFTypeRef> &target, AuthorityType type, CFDictionary &context, std::string *signUnsigned);
	
	void selectRules(SQLite::Statement &action, std::string stanza, std::string table,
		CFTypeRef inTarget, AuthorityType type, SecAssessmentFlags flags, CFDictionaryRef context, std::string suffix = "");
	CFDictionaryRef manipulateRules(const std::string &stanza,
		CFTypeRef target, AuthorityType type, SecAssessmentFlags flags, CFDictionaryRef context, bool authorize);

	void setOrigin(CFArrayRef chain, CFMutableDictionaryRef result);

	void recordOutcome(SecStaticCodeRef code, bool allow, AuthorityType type, double expires, SQLite::int64 authority);

private:
	OpaqueAllowlist* mOpaqueAllowlist;
	CFDictionaryRef opaqueAllowlistValidationConditionsFor(SecStaticCodeRef code);
	bool opaqueAllowlistContains(SecStaticCodeRef code, SecAssessmentFeedback feedback, OSStatus reason);
	void opaqueAllowlistAdd(SecStaticCodeRef code);

    friend class EvaluationManager;
    friend class EvaluationTask;
};


} // end namespace CodeSigning
} // end namespace Security

#endif //_H_POLICYENGINE
