/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 18, 2023.
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
#include "SecTokend.h"
#include "server.h"
#include <securityd_client/ssclient.h>


namespace Security {
namespace Tokend {


using namespace MachPlusPlus;


//
// Support callbacks provided for the implementation
//
static const SCARD_READERSTATE *cbStartupReaderInfo()
{
	return &server->startupReaderState();
}

static const char *cbTokenUid()
{
	return server->tokenUid();
}

static void *cbMalloc(uint32 size)
{
	return malloc(size);
}

static void cbFree(void *ptr)
{
	return free(ptr);
}

static void *cbMallocSensitive(uint32 size)
{
	return malloc(size);
}

static void cbFreeSensitive(void *ptr)
{
	return free(ptr);
}


//
// Vector of support functions passed to implementation
//
static SecTokendSupport supportVector = {
	cbStartupReaderInfo,
	cbTokenUid,
	cbMalloc, cbFree,
	cbMallocSensitive, cbFreeSensitive
};


extern "C" {


//
// The main driver function.
// This is called from the daemon's main() and takes over from there.
//
int SecTokendMain(int argc, const char * argv[],
	const SecTokendCallbacks *callbacks, SecTokendSupport *support)
{
	// first, check interface version and abort if we don't support it
	if (!callbacks) {
		secdebug("tokenlib", "NULL callback structure");
		exit(1);
	}
	if (callbacks->version != kSecTokendCallbackVersion) {
		secdebug("tokenlib", "callback structure is version %d (supporting %d)",
			callbacks->version, kSecTokendCallbackVersion);
		exit(1);
	}
	secdebug("tokenlib", "API interface version %d", callbacks->version);

	server = new Server();
	if (!server)
	{
		secdebug("tokenlib", "can't create server object");
		exit(1);
	}
	
	// set globals (we know by now that the version is okay)
	server->callbacks() = *callbacks;
	if (support)
		*support = supportVector;

	try {
		return (*server)(argc, argv, callbacks->flags);
	} catch (...) {
		secdebug("tokenlib", "server aborted with exception");
		return 1;
	}
}

}	// extern "C"


}	// namespace Tokend
}	// namespace Security
