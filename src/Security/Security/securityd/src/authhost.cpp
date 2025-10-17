/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 4, 2023.
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
#include <paths.h>
#include <fcntl.h>
#include "authhost.h"
#include "server.h"
#include <security_utilities/logging.h>
#include <security_utilities/debugging.h>
#include <bsm/audit.h>
#include <bootstrap_priv.h>

#include <grp.h>
#include <pwd.h>
#include <sys/types.h>
#include <sys/sysctl.h>
#include <syslog.h>
#include <pthread.h>

static pthread_once_t agent_cred_init = PTHREAD_ONCE_INIT; 
static gid_t agent_gid = 92;
static uid_t agent_uid = 92;

static void initialize_agent_creds()
{
    struct passwd *agentUser = getpwnam("securityagent");
    if (agentUser)
    {
        agent_uid = agentUser->pw_uid;
        agent_gid = agentUser->pw_gid;
        endpwent();
    }
}
  
AuthHostInstance::AuthHostInstance(Session &session)
{
	secinfo("authhost", "authhost born (%p)", this);
	referent(session);
	session.addReference(*this);
	pthread_once(&agent_cred_init, initialize_agent_creds);
}

AuthHostInstance::~AuthHostInstance()
{ 
	secinfo("authhost", "authhost died (%p)", this);
}

Session &AuthHostInstance::session() const
{
	return referent<Session>();
}

bool AuthHostInstance::inDarkWake()
{
	return session().server().inDarkWake();
}

void
AuthHostInstance::childAction()
{
	secinfo("AuthHostInstance", "authhostinstance not supported");
	// Unconditional suicide follows.
	_exit(1);
}
