/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 23, 2023.
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
#ifndef _H_PROCESS
#define _H_PROCESS

#include "structure.h"
#include "session.h"
#include <security_utilities/refcount.h>
#include <security_utilities/ccaudit.h>
#include "clientid.h"
#include "localkey.h"
#include "notifications.h"
#include <string>

using MachPlusPlus::Port;
using MachPlusPlus::TaskPort;
using MachPlusPlus::Bootstrap;

class Session;
class LocalDatabase;
class AuthorizationToken;


//
// A Process object represents a UNIX process (and associated Mach Task) that has
// had contact with us and may have some state associated with it. It primarily tracks
// the process nature of the client. Individual threads in the client are tracked by
// Connection objects.
//
// ClientIdentification tracks the identity of guests in the client *as securityd clients*.
// It is concerned with which guest is asking for securityd services, and whether this
// should be granted.
//
class Process : public PerProcess,
				public ClientIdentification{
public:
	Process(TaskPort tPort, Bootstrap bootstrapPort, const ClientSetupInfo *info, const CommonCriteria::AuditToken &audit);
	virtual ~Process();
	
	void reset(TaskPort tPort, const ClientSetupInfo *info, const CommonCriteria::AuditToken &audit);
    
    uid_t uid() const			{ return mUid; }
    gid_t gid() const			{ return mGid; }
    pid_t pid() const			{ return mPid; }
    Security::CommonCriteria::AuditToken const &audit_token() const { return mAudit; }
    TaskPort taskPort() const	{ return mTaskPort; }
    Bootstrap bootstrap() const	{ return mBootstrap; }
	bool byteFlipped() const	{ return mByteFlipped; }
	
	using PerProcess::kill;
	void kill();
	
	void changeSession(Session::SessionId sessionId);
    
	Session& session() const;
	void checkSession(const audit_token_t &auditToken);
	
	LocalDatabase &localStore();
	Key *makeTemporaryKey(const CssmKey &key, CSSM_KEYATTR_FLAGS moreAttributes,
		const AclEntryPrototype *owner);

	// aclSequence is taken to serialize ACL validations to pick up mutual changes
	Mutex aclSequence;

    // Dumping is buggy and only hurts debugging. It's dead Jim.
	//IFDUMP(void dumpNode());
	
private:
	void setup(const ClientSetupInfo *info);
	
private:
	// peer state: established during connection startup; fixed thereafter
    TaskPort mTaskPort;					// task name port
    Bootstrap mBootstrap;				// bootstrap port
	bool mByteFlipped;					// client's byte order is reverse of ours
    pid_t mPid;							// process id
    uid_t mUid;							// UNIX uid credential
    gid_t mGid;							// primary UNIX gid credential

    Security::CommonCriteria::AuditToken const mAudit; // audit token

	// canonical local (transient) key store
	RefPointer<LocalDatabase> mLocalStore;
};


//
// Convenience comparison
//
inline bool operator == (const Process &p1, const Process &p2)
{
	return &p1 == &p2;
}


#endif //_H_PROCESS
