/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 3, 2022.
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
// AuthorizationTrampoline - simple suid-root execution trampoline
// for the authorization API.
//
#include <errno.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <syslog.h>
#include <Security/Authorization.h>
#include <Security/AuthorizationTags.h>
#include <security_utilities/endian.h>
#include <security_utilities/debugging.h>
#include <security_utilities/logging.h>


#define EXECUTERIGHT kAuthorizationRightExecute

//
// A few names for clarity's sake
//
enum {
	READ = 0,		// read end of standard UNIX pipe
	WRITE = 1		// write end of standard UNIX pipe
};

static void fail(OSStatus cause) __attribute__ ((noreturn));


//
// Main program entry point.
//
// Arguments:
//	argv[0] = my name
//	argv[1] = path to user tool
//	argv[2] = "auth n", n=data pipe
//	argv[3..n] = arguments to pass on
//
// File descriptors (set by fork/exec code in client):
//	0 -> communications pipe (perhaps /dev/null)
//	1 -> notify pipe write end
//	2 and above -> unchanged from original client
//
int main(int argc, const char *argv[])
{
	// initial setup
	Syslog::open("authexec", LOG_AUTH);

	// validate basic integrity
	if (!argv[0] || !argv[1] || !argv[2]) {
		Syslog::alert("invalid argument vector");
		exit(1);
	}
	
	// pick up arguments
	const char *pathToTool = argv[1];
	const char *pipeText = argv[2];
	const char **restOfArguments = argv + 3;
	secdebug("authtramp", "trampoline(%s,%s)", pathToTool, mboxFdText);

    // read the external form
    AuthorizationExternalForm extForm;
    int fd;
    if (sscanf(pipeText, "auth %d", &fd) != 1)
        return errAuthorizationInternal;
	ssize_t numOfBytes = read(fd, &extForm, sizeof(extForm));
	close(fd);
    if (numOfBytes != sizeof(extForm)) {
        fail(errAuthorizationInternal);
    }

	// internalize the authorization
	AuthorizationRef auth;
	if (OSStatus error = AuthorizationCreateFromExternalForm(&extForm, &auth)) {
		fail(error);
	}
	secdebug("authtramp", "authorization recovered");
	
	// are we allowed to do this?
	AuthorizationItem right = { EXECUTERIGHT, 0, NULL, 0 };
	AuthorizationRights inRights = { 1, &right };
	AuthorizationRights *outRights;
	if (OSStatus error = AuthorizationCopyRights(auth, &inRights, NULL /*env*/,
			kAuthorizationFlagExtendRights | kAuthorizationFlagInteractionAllowed, &outRights))
		fail(error);
	if (outRights->count != 1 || strcmp(outRights->items[0].name, EXECUTERIGHT))
		fail(errAuthorizationDenied);
		
	// ----- AT THIS POINT WE COMMIT TO PERMITTING THE EXECUTION -----
	
	// let go of our authorization - the client tool will re-internalize it
	AuthorizationFree(auth, kAuthorizationFlagDefaults);

	// make a data pipe
	int dataPipe[2];
	if (pipe(dataPipe)) {
		secinfo("authtramp", "data pipe failure");
		fail(errAuthorizationToolExecuteFailure);
	}

	if (write(dataPipe[WRITE], &extForm, sizeof(extForm)) != sizeof(extForm)) {
		secinfo("authtramp", "fwrite data failed (errno=%d)", errno); // do not fail as only deprecated AuthorizationCopyPrivilegedReference relies on this
	}
	close(dataPipe[WRITE]);

	char pipeFdText[20];
	snprintf(pipeFdText, sizeof(pipeFdText), "auth %d", dataPipe[READ]);

	close(dataPipe[WRITE]);
	// put the external authorization form into the environment
	setenv("__AUTHORIZATION", pipeFdText, true);
	setenv("_BASH_IMPLICIT_DASH_PEE", "-p", true);

	// shuffle file descriptors
	int notify = dup(1);		// save notify port
	fcntl(notify, F_SETFD, 1);	// close notify port on (successful) exec
	dup2(0, 1);					// make stdin, stdout point to the comms pipe
	
	// prepare the argv for the tool (prepend the "myself" element)
	// note how this overwrites a known-existing argv element (that we copied earlier)
	*(--restOfArguments) = pathToTool;
	
	secdebug("authtramp", "trampoline executes %s", pathToTool);
	Syslog::notice("executing %s", pathToTool);
	execv(pathToTool, (char *const *)restOfArguments);
	secdebug("authexec", "exec(%s) failed (errno=%d)", pathToTool, errno);
	
	// report failure
	OSStatus error = h2n(OSStatus(errAuthorizationToolExecuteFailure));
	write(notify, &error, sizeof(error));
	exit(1);
}


void fail(OSStatus cause)
{
	OSStatus tmp = h2n(cause);
	write(1, &tmp, sizeof(tmp));	// ignore error - can't do anything if error
	secinfo("authtramp", "trampoline aborting with status %d", (int)cause);
	exit(1);
}
