/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 3, 2023.
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
#include <strings.h>	// bcopy()
#include <errno.h>
#include <bsm/audit_record.h>
#include <bsm/libbsm.h>
#include <security_utilities/utilities.h>
#include <security_utilities/debugging.h>
#include <security_utilities/logging.h>
#include <security_utilities/errors.h>
#include <security_utilities/ccaudit.h>

namespace Security
{
namespace CommonCriteria
{

TerminalId::TerminalId()
{
	if (audit_set_terminal_id(this) != kAUNoErr)
	{
		Syslog::warning("setting terminal ID info failed; using defaults");
		port = 0;
		machine = 0;
	}
}

AuditToken::AuditToken(const audit_token_t &token)
    : mAuditToken(token)
{
    ::audit_token_to_au32(token, &mAuditId, &mEuid, &mEgid, &mRuid, &mRgid, &mPid, &mSessionId, &mTerminalId);
}


//
// AuditInfo
//
void AuditInfo::get()
{
	this->clearPod();
	UnixError::check(::getaudit_addr(this, sizeof(*this)));
}

void AuditInfo::get(au_asid_t session)
{
	this->get();
	if (session != this->ai_asid) {
		// need to use higher-privileged call to get info about a session that is not our own
		this->ai_asid = session;
		UnixError::check(::auditon(A_GETSINFO_ADDR, this, sizeof(*this)));
	}
}

void AuditInfo::getPid(pid_t pid)
{
	auditpinfo_addr_t pinfo;
	memset(&pinfo, 0, sizeof(pinfo));
	pinfo.ap_pid = pid;
	UnixError::check(::auditon(A_GETPINFO_ADDR, &pinfo, sizeof(pinfo)));
	get(pinfo.ap_asid);
}

void AuditInfo::set()
{
	UnixError::check(::setaudit_addr(this, sizeof(*this)));
}

void AuditInfo::create(uint64_t flags, uid_t auid /* = AU_DEFAUDITID */)
{
	this->clearPod();
	ai_auid = auid;
	ai_asid = AU_ASSIGN_ASID;
	ai_termid.at_type = AU_IPv4;
	ai_flags = flags;
	UnixError::check(::setaudit_addr(this, sizeof(*this)));
}


void AuditSession::registerSession(void)
{
    auditinfo_t auinfo;

    auinfo.ai_auid = mAuditId;
    auinfo.ai_asid = mSessionId;
    bcopy(&mTerminalId, &(auinfo.ai_termid), sizeof(auinfo.ai_termid));
    bcopy(&mEventMask.get(), &(auinfo.ai_mask), sizeof(auinfo.ai_mask));

    if (setaudit(&auinfo) != 0)
	{
		if (errno == ENOTSUP)
			Syslog::notice("Attempted to initialize auditing, but this kernel does not support auditing");
		else
			Syslog::warning("Could not initialize auditing (%m); continuing");
	}
}

void AuditRecord::submit(const short event_code, const int returnCode, 
			 const char *msg)
{
    // If we're not auditing, do nothing
    if (!(au_get_state() == AUC_AUDITING))
		return;

    secinfo("ccaudit", "Submitting authorization audit record");

    int ret = kAUNoErr;

    // XXX/gh  3574731: Fix BSM SPI so the const_cast<>s aren't necessary
    if (returnCode == 0)
    {
		token_t *tok = NULL;

		if (msg)
			tok = au_to_text(const_cast<char *>(msg));
		ret = audit_write_success(event_code, const_cast<token_t *>(tok), 
								  mAuditToken.auditId(), mAuditToken.euid(),
								  mAuditToken.egid(), mAuditToken.ruid(), 
								  mAuditToken.rgid(), mAuditToken.pid(), 
								  mAuditToken.sessionId(),
								  const_cast<au_tid_t *>(&(mAuditToken.terminalId())));
    }
    else
    {
		ret = audit_write_failure(event_code, const_cast<char *>(msg), 
								  returnCode, mAuditToken.auditId(), 
								  mAuditToken.euid(), mAuditToken.egid(), 
								  mAuditToken.ruid(), mAuditToken.rgid(), 
								  mAuditToken.pid(), mAuditToken.sessionId(),
								  const_cast<au_tid_t *>(&(mAuditToken.terminalId())));
    }
    if (ret != kAUNoErr)
		MacOSError::throwMe(ret);
}


}	// end namespace CommonCriteria
}	// end namespace Security
