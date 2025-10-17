/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 25, 2022.
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
// Authorization.cpp
//
// This file is the unified implementation of the Authorization and AuthSession APIs.
//
#include <stdint.h>
#include <Security/AuthSession.h>
#include <Security/AuthorizationPriv.h>
#include <security_utilities/ccaudit.h>
#include <security_cdsa_utilities/cssmbridge.h>
#include <Security/SecBase.h>
#include <security_utilities/logging.h>
#include "LegacyAPICounts.h"

//
// This no longer talks to securityd; it is a kernel function.
//
OSStatus SessionGetInfo(SecuritySessionId requestedSession,
    SecuritySessionId *sessionId,
    SessionAttributeBits *attributes)
{
    BEGIN_API_NO_METRICS
    if (requestedSession != noSecuritySession && requestedSession != callerSecuritySession) {
        static dispatch_once_t countToken;
        countLegacyAPI(&countToken, __FUNCTION__);
    }
	CommonCriteria::AuditInfo session;
	if (requestedSession == callerSecuritySession)
		session.get();
	else
		session.get(requestedSession);
	if (sessionId)
		*sessionId = session.sessionId();
	if (attributes)
        *attributes = (SessionAttributeBits)session.flags();
    END_API_NO_METRICS(CSSM)
}


//
// Create a new session.
// This no longer talks to securityd; it is a kernel function.
// Securityd will pick up the new session when we next talk to it.
//
OSStatus SessionCreate(SessionCreationFlags flags,
    SessionAttributeBits attributes)
{
    BEGIN_API

	// we don't support the session creation flags anymore
	if (flags)
		Syslog::warning("SessionCreate flags=0x%lx unsupported (ignored)", (unsigned long)flags);
	CommonCriteria::AuditInfo session;
	session.create(attributes);
        
	// retrieve the (new) session id and set it into the process environment
	session.get();
	char idString[80];
	snprintf(idString, sizeof(idString), "%x", session.sessionId());
	setenv("SECURITYSESSIONID", idString, 1);

    END_API(CSSM)
}


//
// Get and set the distinguished uid (optionally) associated with the session.
//
OSStatus SessionSetDistinguishedUser(SecuritySessionId session, uid_t user)
{
	BEGIN_API
	CommonCriteria::AuditInfo session;
	session.get();
	session.ai_auid = user;
	session.set();
	END_API(CSSM)
}


OSStatus SessionGetDistinguishedUser(SecuritySessionId session, uid_t *user)
{
    BEGIN_API
	CommonCriteria::AuditInfo session;
	session.get();
	Required(user) = session.uid();
    END_API(CSSM)
}

OSStatus SessionSetUserPreferences(SecuritySessionId session)
{
    return errSecSuccess;
}
