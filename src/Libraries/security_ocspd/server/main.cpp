/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 25, 2023.
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
/*
 * main.cpp - main() for OCSP helper daemon
 */

#include <stdlib.h>
#include "ocspdServer.h"
#include <security_ocspd/ocspdTypes.h>
#include <security_ocspd/ocspdDebug.h>
#include <security_utilities/daemon.h>
#include <security_utilities/logging.h>
#include <CoreFoundation/CoreFoundation.h>

using namespace Security;

Mutex gTimeMutex;

const CFAbsoluteTime kTimeoutInterval = 300;
const int kTimeoutCheckTime = 60;

extern void enableAutoreleasePool(int enable);

static void usage(char **argv)
{
	printf("Usage: %s [option...]\n", argv[0]);
	printf("Options:\n");
	printf("  -d                  -- Debug mode, do not run as forked daemon\n");
	printf("  -n bootstrapName    -- specify alternate bootstrap name\n");
	exit(1);
}

void HandleSigTerm (int sig)
{
	exit (1);
}

CFAbsoluteTime gLastActivity;

void ServerActivity()
{
	StLock<Mutex> _mutexLock(gTimeMutex);
	gLastActivity = CFAbsoluteTimeGetCurrent();
}

class TimeoutTimer : public MachPlusPlus::MachServer::Timer
{
protected:
	OcspdServer &mServer;

public:
	TimeoutTimer(OcspdServer &server) : mServer(server) {}
	void action();
};


void TimeoutTimer::action()
{
	bool doExit = false;
	{
		StLock<Mutex> _mutexLock(gTimeMutex);
		CFAbsoluteTime thisTime = CFAbsoluteTimeGetCurrent();
		if (thisTime - gLastActivity > kTimeoutInterval)
		{
			doExit = true;
		}
	
		if (!doExit)
		{
			// reinstall us as a timer
			mServer.setTimer(this, Time::Interval(kTimeoutCheckTime));
		}
	}

	if (doExit)
	{
		exit(0);
	}
}

int main(int argc, char **argv)
{
/* ****************************************************************************
 * IMPORTANT: The functionality provided by the OCSP helper daemon (ocspd)
 * has been subsumed by trustd as of macOS 12.0 (Monterey). SecTrust APIs
 * no longer rely on the ocspd daemon, and CDSA APIs have been deprecated
 * for the past decade. The daemon is currently disabled and will be removed.
 * ****************************************************************************
 */
	Syslog::alert("ocspd is disabled and not intended to be invoked directly.");
	exit(1);
#if 0
	signal (SIGTERM, HandleSigTerm);
	enableAutoreleasePool(1);

	/* user-specified variables */
	const char *bootStrapName = NULL;
	bool debugMode = false;
	
	extern char *optarg;
	int arg;
	while ((arg = getopt(argc, argv, "dn:h")) != -1) {
		switch(arg) {
			case 'd':
				debugMode = true;
				break;
			case 'n':
				bootStrapName = optarg;
				break;
			case 'h':
			default:
				usage(argv);
		}
	}
	
	/* no non-option arguments */
	if (optind < argc) {
		usage(argv);
	}
	
	/* bootstrap name override for debugging only */
	#ifndef	NDEBUG
	if(bootStrapName == NULL) {
		bootStrapName = getenv(OCSPD_BOOTSTRAP_ENV);
	}
	#endif	/* NDEBUG */
	if(bootStrapName == NULL) {
		bootStrapName = OCSPD_BOOTSTRAP_NAME;
	}
	
    /* if we're not running as root in production mode, fail */
	#if defined(NDEBUG)
    if (uid_t uid = getuid()) {
        Syslog::alert("Tried to run ocspd as user %d: aborted", uid);
        fprintf(stderr, "You are not allowed to run securityd\n");
        exit(1);
    }
	#endif //NDEBUG

    /* turn into a properly diabolical daemon unless debugMode is on */
    if (!debugMode) {
		if (!Daemon::incarnate(false))
			exit(1);	// can't daemonize
	}

    // Declare the server here.  That way if something throws underneath its state wont
    // fall out of scope, taking the server global state with it.  That will let us shut
    // down more peacefully.
	OcspdServer server(bootStrapName);

	try {
		/* create the main server object and register it */

		/* FIXME - any signal handlers? */

		ocspdDebug("ocspd: starting main run loop");

		ServerActivity();
		TimeoutTimer tt(server);
		/* These options copied from securityd - they enable the audit trailer */
		server.setTimer(&tt, Time::Interval(kTimeoutCheckTime));

		server.run(4096,		// copied from machserver default
			MACH_RCV_TRAILER_TYPE(MACH_MSG_TRAILER_FORMAT_0) |
			MACH_RCV_TRAILER_ELEMENTS(MACH_RCV_TRAILER_AUDIT));
	}
	catch(...) {}
	/* fell out of runloop (should not happen) */
	enableAutoreleasePool(0);
	#ifndef NDEBUG
	Syslog::alert("Aborting");
	#endif
	return 1;
#endif
}
