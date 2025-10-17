/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 14, 2022.
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
// CodeSigner - SecCodeSigner API objects
//
#ifndef _H_CODESIGNER
#define _H_CODESIGNER

#include "cs.h"
#include "StaticCode.h"
#include "cdbuilder.h"
#include <Security/SecIdentity.h>
#include <security_utilities/utilities.h>

namespace Security {
namespace CodeSigning {


//
// SecCodeSigner is responsible for signing code objects
//
class SecCodeSigner : public SecCFObject {
	NOCOPY(SecCodeSigner)
public:
	class Parser;
	class Signer;

public:
	SECCFFUNCTIONS(SecCodeSigner, SecCodeSignerRef, errSecCSInvalidObjectRef, gCFObjects().CodeSigner)

	SecCodeSigner(SecCSFlags flags);
    virtual ~SecCodeSigner() _NOEXCEPT;
	
	void parameters(CFDictionaryRef args);	// parse and set parameters
	virtual bool valid() const;
    
	std::string getTeamIDFromSigner(CFArrayRef certs);
	
	void sign(SecStaticCode *code, SecCSFlags flags);
	void remove(SecStaticCode *code, SecCSFlags flags);
	
	void returnDetachedSignature(BlobCore *blob, Signer &signer);
	
	const CodeDirectory::HashAlgorithms &digestAlgorithms() const { return mDigestAlgorithms; }
	
public:
	// parsed parameter set
	SecCSFlags mOpFlags;			// operation flags
	CFRef<SecIdentityRef> mSigner;	// signing identity
	CFRef<CFTypeRef> mDetached;		// detached-signing information (NULL => attached)
	CFRef<CFDictionaryRef> mResourceRules; // explicit resource collection rules (override)
	CFRef<CFDateRef> mSigningTime;	// signing time desired (kCFNull for none)
	CFRef<CFDataRef> mApplicationData; // contents of application slot
	CFRef<CFDataRef> mEntitlementData; // entitlement configuration data
	bool mForceLibraryEntitlements; // force entitlements when signing libraries
	vector<CFRef<CFDataRef>> mLaunchConstraints; // Array of Lightweight Code Requirements
	CFRef<CFDataRef> mLibraryConstraints; // Library load Lightweight Code Requirement
	CFRef<CFURLRef> mSDKRoot;		// substitute filesystem root for sub-component lookup
	CFRef<CFTypeRef> mRequirements; // internal code requirements
	size_t mCMSSize;				// size estimate for CMS blob
	uint32_t mCdFlags;				// CodeDirectory flags
	uint32_t mPreserveMetadata;		// metadata preservation options
	bool mCdFlagsGiven;				// CodeDirectory flags were specified
	CodeDirectory::HashAlgorithms mDigestAlgorithms; // interior digest (hash) algorithm
	std::string mIdentifier;		// unique identifier override
	std::string mIdentifierPrefix;	// prefix for un-dotted default identifiers
	std::string mTeamID;            // teamID
	PlatformIdentifier mPlatform;	// platform identifier (zero if not platform binary)
	bool mNoMachO;					// override to perform non-Mach-O signing
	bool mDryRun;					// dry run (do not change target)
	CFRef<CFNumberRef> mPageSize;	// main executable page size
	CFRef<SecIdentityRef> mTimestampAuthentication;	// identity for client-side authentication to the Timestamp server
	CFRef<CFURLRef> mTimestampService;		// URL for Timestamp server
    bool mWantTimeStamp;          // use a Timestamp server
    bool mNoTimeStampCerts;       // don't request certificates with timestamping request
	LimitedAsync *mLimitedAsync;	// limited async workers for verification
	uint32_t mRuntimeVersionOverride;	// runtime Version Override
	bool mPreserveAFSC;             // preserve AFSC compression
	bool mOmitAdhocFlag;			// don't add adhoc flag, even without signer identity

	// Signature Editing
	Architecture mEditArch;			// architecture to edit (defaults to all if empty)
	CFRef<CFDataRef> mEditCMS;		// CMS to replace in the signature
	
};


} // end namespace CodeSigning
} // end namespace Security

#endif // !_H_CODESIGNER
