/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 26, 2024.
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
#include "csprocess.h"
#include "cskernel.h"
#include <securityd_client/ssclient.h>
#include <System/sys/codesign.h>
#include "LWCRHelper.h"

namespace Security {
namespace CodeSigning {


//
// Construct a running process representation
//
ProcessCode::ProcessCode(pid_t pid, const audit_token_t* token, PidDiskRep *pidDiskRep /*= NULL */)
	: SecCode(KernelCode::active()), mPid(pid), mPidBased(pidDiskRep)
{
	if (token)
		mAudit = new audit_token_t(*token);
	else
		mAudit = NULL;
}


int ProcessCode::csops(unsigned int ops, void *addr, size_t size)
{
	// pass pid and audit token both if we have it, or just the pid if we don't
	if (mAudit)
		return ::csops_audittoken(mPid, ops, addr, size, mAudit);
	else
		return ::csops(mPid, ops, addr, size);
}

void ProcessCode::codeMatchesLightweightCodeRequirementData(CFDataRef lwcrData)
{
#if !TARGET_OS_SIMULATOR
	if (mAudit) {
		evaluateLightweightCodeRequirementInKernel(*mAudit, lwcrData);
	} else {
		MacOSError::throwMe(errSecParam);
	}
#else
	MacOSError::throwMe(errSecCSUnimplemented)
#endif

}


/*
 *
 */
        
ProcessDynamicCode::ProcessDynamicCode(ProcessCode *guest)
        : SecStaticCode(guest->pidBased()), mGuest(guest)
{
}

CFDataRef ProcessDynamicCode::component(CodeDirectory::SpecialSlot slot, OSStatus fail /* = errSecCSSignatureFailed */)
{
        if (slot == cdInfoSlot && !mGuest->pidBased()->supportInfoPlist())
                return NULL;
        else if (slot == cdResourceDirSlot)
                return NULL;
        return SecStaticCode::component(slot, fail);
}

CFDictionaryRef ProcessDynamicCode::infoDictionary()
{
        if (mGuest->pidBased()->supportInfoPlist())
                return SecStaticCode::infoDictionary();
        if (!mEmptyInfoDict) {
                mEmptyInfoDict.take(makeCFDictionary(0));
        }
        return mEmptyInfoDict;
}

void ProcessDynamicCode::validateComponent(CodeDirectory::SpecialSlot slot, OSStatus fail /* = errSecCSSignatureFailed */)
{
        if (slot == cdInfoSlot && !mGuest->pidBased()->supportInfoPlist())
                return;
        else if (slot == cdResourceDirSlot)
                return;
        SecStaticCode::validateComponent(slot, fail);
}


        
} // CodeSigning
} // Security
