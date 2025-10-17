/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 10, 2025.
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
/*!
	@header CSCommonPriv
	SecStaticCodePriv is the private counter-part to CSCommon. Its contents are not
	official API, and are subject to change without notice.
*/
#ifndef _H_CSCOMMONPRIV
#define _H_CSCOMMONPRIV

#include <Security/CSCommon.h>

#ifdef __cplusplus
extern "C" {
#endif


/*!
	@typedef SecCodeDirectoryFlagTable
	This constant array can be used to translate between names and values
	of CodeDirectory flag bits. The table ends with an entry with NULL name.
	The elements are in no particular order.
	@field name The official text name of the flag.
	@field value The binary value of the flag.
	@field signable True if the flag can be specified during signing. False if it is set
	internally and can only be read from a signature.
 */
typedef struct {
	const char *name;
	uint32_t value;
	bool signable;
} SecCodeDirectoryFlagTable;

extern const SecCodeDirectoryFlagTable kSecCodeDirectoryFlagTable[];


/*!
	Blob types (magic numbers) for blobs used by Code Signing.
	
	@constant kSecCodeMagicRequirement Magic number for individual code requirements.
	@constant kSecCodeMagicRequirementSet Magic number for a collection of
	individual code requirements, indexed by requirement type. This is used
	for internal requirement sets.
	@constant kSecCodeMagicCodeDirectory Magic number for a CodeDirectory.
	@constant kSecCodeMagicEmbeddedSignature Magic number for a SuperBlob
	containing all the signing components that are usually embedded within
	a main executable.
	@constant kSecCodeMagicDetachedSignature Magic number for a SuperBlob that
	contains all the data for all architectures of a signature, including any
	data that is usually written to separate files. This is the format of
	detached signatures if the program is capable of having multiple architectures.
	@constant kSecCodeMagicEntitlement Magic number for a standard entitlement blob.
 	@constant kSecCodeMagicEntitlementDER Magic number for a DER entitlement blob.
    @constant kSecCodeMagicLaunchConstraint Magic number for the launch constraints blobs.
	@constant kSecCodeMagicByte The first byte (in NBO) shared by all these magic
	numbers. This is not a valid ASCII character; test for this to distinguish
	between text and binary data if you expect a code signing-related binary blob.
 */

enum {
	kSecCodeMagicRequirement = 0xfade0c00,		/* single requirement */
	kSecCodeMagicRequirementSet = 0xfade0c01,	/* requirement set */
	kSecCodeMagicCodeDirectory = 0xfade0c02,	/* CodeDirectory */
	kSecCodeMagicEmbeddedSignature = 0xfade0cc0, /* single-architecture embedded signature */
	kSecCodeMagicDetachedSignature = 0xfade0cc1, /* detached multi-architecture signature */
	kSecCodeMagicEntitlement = 0xfade7171,		/* entitlement blob */
	kSecCodeMagicEntitlementDER = 0xfade7172,	/* entitlement DER blob */
    kSecCodeMagicLaunchConstraint = 0xfade8181, /* all of the launch constraints */
	kSecCodeMagicByte = 0xfa					/* shared first byte */
};

/*!
 @typedef SecCodeExecSegFlags
 */
typedef CF_OPTIONS(uint32_t, SecCodeExecSegFlags) {
	kSecCodeExecSegMainBinary = 0x0001,		/* exec seg belongs to main binary */

	// Entitlements
	kSecCodeExecSegAllowUnsigned = 0x0010,	/* allow unsigned pages (for debugging) */
	kSecCodeExecSegDebugger = 0x0020,		/* main binary is debugger */
	kSecCodeExecSegJit = 0x0040,			/* JIT enabled */
	kSecCodeExecSegSkipLibraryVal = 0x0080,	/* skip library validation */
	kSecCodeExecSegCanLoadCdHash = 0x0100,	/* can bless cdhash for execution */
	kSecCodeExecSegCanExecCdHash = 0x0200,	/* can execute blessed cdhash */
};
	
/*
	The current (fixed) size of a cdhash in the system.
 */
enum {
	kSecCodeCDHashLength = 20
};


/*!
	A callback block type for monitoring certain code signing operations
 */
typedef CFTypeRef (^SecCodeCallback)(SecStaticCodeRef code, CFStringRef stage, CFDictionaryRef info);


#ifdef __cplusplus
}
#endif

#endif //_H_CSCOMMON
