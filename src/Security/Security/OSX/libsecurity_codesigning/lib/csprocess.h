/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 23, 2022.
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
// csprocess - UNIX process implementation of the Code Signing Host Interface
//
#ifndef _H_CSPROCESS
#define _H_CSPROCESS

#include "Code.h"
#include "StaticCode.h"
#include "piddiskrep.h"
#include <TargetConditionals.h>
#include <security_utilities/utilities.h>

namespace Security {
namespace CodeSigning {


//
// A SecCode that represents a running UNIX process.
// Processes are identified by pid and audit token.
//
class ProcessCode : public SecCode {
public:
	ProcessCode(pid_t pid, const audit_token_t* token, PidDiskRep *pidDiskRep = NULL);
	~ProcessCode() _NOEXCEPT { delete mAudit; }
	
	pid_t pid() const { return mPid; }
	const audit_token_t* audit() const { return mAudit; }
	
	PidDiskRep *pidBased() const { return mPidBased; }
	
	int csops(unsigned int ops, void *addr, size_t size);
	void codeMatchesLightweightCodeRequirementData(CFDataRef lwcrData);

private:
	pid_t mPid;
	audit_token_t* mAudit;
	RefPointer<PidDiskRep> mPidBased;
};


//
// We don't need a GenericCode variant of ProcessCode
//
typedef SecStaticCode ProcessStaticCode;
        
class ProcessDynamicCode : public SecStaticCode {
public:
	ProcessDynamicCode(ProcessCode *diskRep);

        CFDataRef component(CodeDirectory::SpecialSlot slot, OSStatus fail = errSecCSSignatureFailed);
        
        CFDictionaryRef infoDictionary();
        
        void validateComponent(CodeDirectory::SpecialSlot slot, OSStatus fail = errSecCSSignatureFailed);
private:
        ProcessCode *mGuest;
        CFRef<CFDictionaryRef> mEmptyInfoDict;
};

} // end namespace CodeSigning
} // end namespace Security

#endif // !_H_CSPROCESS
