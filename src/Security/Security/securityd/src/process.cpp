/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 3, 2024.
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
// process - track a single client process and its belongings
//
#include "process.h"
#include "server.h"
#include "session.h"
#include "tempdatabase.h"
#include "child.h"          // ServerChild (really UnixPlusPlus::Child)::find()

#include <security_utilities/ccaudit.h>
#include <security_utilities/logging.h>	//@@@ debug only
#include "agentquery.h"


//
// Construct a Process object.
//
Process::Process(TaskPort taskPort, Bootstrap bootstrapPort, const ClientSetupInfo *info, const CommonCriteria::AuditToken &audit)
 :  mTaskPort(taskPort), mBootstrap(bootstrapPort), mByteFlipped(false), mPid(audit.pid()), mUid(audit.euid()), mGid(audit.egid()), mAudit(audit)
{
	StLock<Mutex> _(*this);
    xpc_transaction_begin();
	// set parent session
	parent(Session::find(audit.sessionId(), true));
	
	// let's take a look at our wannabe client...
	
	// Not enough to make sure we will get the right process, as
	// pids get recycled. But we will later create the actual SecCode using
	// the audit token, which is unique to the one instance of the process,
	// so this just catches a pid mismatch early.
	if (mTaskPort.pid() != mPid) {
		secnotice("SecServer", "Task/pid setup mismatch pid=%d task=%d(%d)",
				  mPid, mTaskPort.port(), mTaskPort.pid());
		CssmError::throwMe(CSSMERR_CSSM_ADDIN_AUTHENTICATE_FAILED);	// you lied!
	}
	
	setup(info);
	ClientIdentification::setup(this->audit_token());
	
	if(!processCode()) {
		// This can happen if the process died in the meantime.
		secnotice("SecServer", "no process created in setup, old pid=%d old task=%d(%d)",
				  mPid, mTaskPort.port(), mTaskPort.pid());
		CssmError::throwMe(CSSMERR_CSSM_ADDIN_AUTHENTICATE_FAILED);
	}
	
	// This is a "retain", matched by the deallocate call in ~Process
	mTaskPort.modRefs(MACH_PORT_RIGHT_SEND, 1);
	mBootstrap.modRefs(MACH_PORT_RIGHT_SEND, 1);

    // NB: ServerChild::find() should only be used to determine
    // *existence*.  Don't use the returned Child object for anything else, 
    // as it is not protected against its underlying process's destruction.  
	if (this->pid() == getpid() // called ourselves (through some API). Do NOT record this as a "dirty" transaction
        || ServerChild::find<ServerChild>(this->pid()))   // securityd's child; do not mark this txn dirty
        xpc_transaction_end();

    secinfo("SecServer", "%p client new: pid:%d session:%d %s taskPort:%d bootstrapPort:%d uid:%d gid:%d", this, this->pid(), this->session().sessionId(),
             (char *)codePath(this->processCode()).c_str(), taskPort.port(), bootstrapPort.port(), mUid, mGid);
}


//
// Screen a process setup request for an existing process.
// This means the client has requested intialization even though we remember having
// talked to it in the past. This could either be an exec(2), or the client could just
// have forgotten all about its securityd client state. Or it could be an attack...
//
void Process::reset(TaskPort taskPort, const ClientSetupInfo *info, const CommonCriteria::AuditToken &audit)
{
	StLock<Mutex> _(*this);
	if (taskPort != mTaskPort) {
		secnotice("SecServer", "Process %p(%d) reset mismatch (tp %d-%d)",
			this, pid(), taskPort.port(), mTaskPort.port());
		//@@@ CssmError::throwMe(CSSM_ERRCODE_VERIFICATION_FAILURE);		// liar
	}
	setup(info);
	CFCopyRef<SecCodeRef> oldCode = processCode();

	ClientIdentification::setup(this->audit_token());	// re-constructs processCode()
	if (CFEqual(oldCode, processCode())) {
        secnotice("SecServer", "%p Client reset amnesia", this);
	} else {
        secnotice("SecServer", "%p Client reset full", this);
	}
}


//
// Common set processing
//
void Process::setup(const ClientSetupInfo *info)
{
	// process setup info
	assert(info);
	uint32 pversion;
	if (info->order == 0x1234) {	// right side up
		pversion = info->version;
		mByteFlipped = false;
	} else if (info->order == 0x34120000) { // flip side up
		pversion = flip(info->version);
		mByteFlipped = true;
	} else // non comprende
		CssmError::throwMe(CSSM_ERRCODE_INCOMPATIBLE_VERSION);

	// check wire protocol version
	if (pversion != SSPROTOVERSION)
		CssmError::throwMe(CSSM_ERRCODE_INCOMPATIBLE_VERSION);
}


//
// Clean up a Process object
//
Process::~Process()
{
    secinfo("SecServer", "%p client release: %d", this, this->pid());

    // release our name for the process's task port
    if (mTaskPort) {
        mTaskPort.deallocate();
    }
    if (mBootstrap) {
        mBootstrap.deallocate();
    }
    xpc_transaction_end();
}

void Process::kill()
{
	StLock<Mutex> _(*this);
	
	// release local temp store
	mLocalStore = NULL;

	// standard kill processing
	PerProcess::kill();
}


Session& Process::session() const
{
	return parent<Session>();
}


void Process::checkSession(const audit_token_t &auditToken)
{
	Security::CommonCriteria::AuditToken audit(auditToken);
	if (audit.sessionId() != this->session().sessionId())
		this->changeSession(audit.sessionId());
}


LocalDatabase &Process::localStore()
{
	StLock<Mutex> _(*this);
	if (!mLocalStore)
		mLocalStore = new TempDatabase(*this);
	return *mLocalStore;
}

Key *Process::makeTemporaryKey(const CssmKey &key, CSSM_KEYATTR_FLAGS moreAttributes,
	const AclEntryPrototype *owner)
{
	return safer_cast<TempDatabase&>(localStore()).makeKey(key, moreAttributes, owner);
}


//
// Change the session of a process.
// This is the result of SessionCreate from a known process client.
//
void Process::changeSession(Session::SessionId sessionId)
{
	// re-parent
	parent(Session::find(sessionId, true));
    secnotice("SecServer", "%p client change session to %d", this, this->session().sessionId());
}


//
// Debug dump support
//
#if defined(DEBUGDUMP)

void Process::dumpNode()
{
	PerProcess::dumpNode();
	if (mByteFlipped)
		Debug::dump(" FLIPPED");
	Debug::dump(" task=%d pid=%d uid/gid=%d/%d",
		mTaskPort.port(), mPid, mUid, mGid);
	ClientIdentification::dump();
}

#endif //DEBUGDUMP
