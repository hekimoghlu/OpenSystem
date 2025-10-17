/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 5, 2025.
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
// demon - support code for writing UNIXoid demons
//
#include <security_utilities/daemon.h>
#include <security_utilities/logging.h>
#include <security_utilities/debugging.h>
#include <sys/types.h>
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>

namespace Security {
namespace Daemon {

//
// Daemonize this process, the UNIX way.
//
bool incarnate(bool doFork /*=true*/)
{
	if (doFork) {
		// fork with slight resilience
		for (int forkTries = 1; forkTries <= 5; forkTries++) {
			switch (fork()) {
				case 0:			// child
							// we are the daemon process (Har! Har!)
					break;
				case -1:		// parent: fork failed
					switch (errno) {
						case EAGAIN:
						case ENOMEM:
							Syslog::warning("fork() short on resources (errno=%d); retrying", errno);
							sleep(forkTries);
							continue;
						default:
							Syslog::error("fork() failed (errno=%d)", errno);
							return false;
					}
				default:		// parent
							// @@@ we could close an assurance loop here, but we don't (yet?)
					exit(0);
			}
		}
		// fork succeeded; we are the child; parent is terminating
	}
	
	// create new session (the magic set-me-apart system call)
	setsid();

	// redirect standard channels to /dev/null
	close(0);	// fail silently in case 0 is closed
	if (open("/dev/null", O_RDWR, 0) == 0) {	// /dev/null could be missing, I suppose...
		dup2(0, 1);
		dup2(0, 2);
	}
	
	// ready to roll
	return true;
}

} // end namespace Daemon
} // end namespace Security
