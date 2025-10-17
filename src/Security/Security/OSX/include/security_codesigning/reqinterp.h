/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 10, 2023.
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
// reqinterp - Requirement language (exprOp) interpreter
//
#ifndef _H_REQINTERP
#define _H_REQINTERP

#include "reqreader.h"
#include <Security/SecTrustSettings.h>

#if TARGET_OS_OSX
#include <security_cdsa_utilities/cssmdata.h>	// CssmOid
#endif

namespace Security {
namespace CodeSigning {


//
// An interpreter for exprForm-type requirements.
// This is a simple Polish Notation stack evaluator.
//	
class Requirement::Interpreter : public Requirement::Reader {	
public:
	Interpreter(const Requirement *req, const Context *ctx)	: Reader(req), mContext(ctx) { }
	
	static const unsigned stackLimit = 1000;
	
	bool evaluate();
	
protected:
	class Match {
	public:
		Match(Interpreter &interp);		// reads match postfix from interp
		Match(CFStringRef value, MatchOperation op) : mValue(value), mOp(op) { } // explicit
		Match() : mValue(NULL), mOp(matchExists) { } // explict test for presence
		bool operator () (CFTypeRef candidate) const; // match to candidate

	protected:
		bool inequality(CFTypeRef candidate, CFStringCompareFlags flags, CFComparisonResult outcome, bool negate) const;
		
	private:
		CFCopyRef<CFTypeRef> mValue;	// match value
		MatchOperation mOp;				// type of match
		
		bool isStringValue() const { return CFGetTypeID(mValue) == CFStringGetTypeID(); }
		bool isDateValue() const { return CFGetTypeID(mValue) == CFDateGetTypeID(); }
		CFStringRef cfStringValue() const { return isStringValue() ? (CFStringRef)mValue.get() : NULL; }
		CFDateRef cfDateValue() const { return isDateValue() ? (CFDateRef)mValue.get() : NULL; }
	};
	
protected:
	bool eval(int depth);
	
	bool infoKeyValue(const std::string &key, const Match &match);
	bool entitlementValue(const std::string &key, const Match &match);
	bool certFieldValue(const string &key, const Match &match, SecCertificateRef cert);
#if TARGET_OS_OSX
	bool certFieldGeneric(const string &key, const Match &match, SecCertificateRef cert);
	bool certFieldGeneric(const CssmOid &oid, const Match &match, SecCertificateRef cert);
	bool certFieldPolicy(const string &key, const Match &match, SecCertificateRef cert);
	bool certFieldPolicy(const CssmOid &oid, const Match &match, SecCertificateRef cert);
	bool certFieldDate(const string &key, const Match &match, SecCertificateRef cert);
	bool certFieldDate(const CssmOid &oid, const Match &match, SecCertificateRef cert);
#endif
	bool verifyAnchor(SecCertificateRef cert, const unsigned char *digest);
	bool appleSigned();
	bool appleAnchored();
	bool inTrustCache();

	bool trustedCerts();
	bool trustedCert(int slot);
	
	static SecTrustSettingsResult trustSetting(SecCertificateRef cert, bool isAnchor);
	
private:
    CFArrayRef getAdditionalTrustedAnchors();
    bool appleLocalAnchored();
	const Context * const mContext;
};


}	// CodeSigning
}	// Security

#endif //_H_REQINTERP
