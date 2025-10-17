/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 19, 2024.
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
// pcscmonitor - use PCSC to monitor smartcard reader/card state for securityd
//
#ifndef _H_PCSCMONITOR
#define _H_PCSCMONITOR

#include "server.h"
#include "tokencache.h"
#include "reader.h"
#include "token.h"
#include <security_utilities/pcsc++.h>
#include <security_utilities/coderepository.h>
#include <set>


//
// A PCSCMonitor uses PCSC to monitor the state of smartcard readers and
// tokens (cards) in the system, and dispatches messages and events to the
// various related players in securityd. There should be at most one of these
// objects active within securityd.
//
class PCSCMonitor : private Listener, private MachServer::Timer {
public:
	enum ServiceLevel {
		forcedOff,					// no service under any circumstances
		externalDaemon				// use externally launched daemon if present (do not manage pcscd)
	};

	PCSCMonitor(Server &server, const char* pathToCache, ServiceLevel level = externalDaemon);

protected:
	Server &server;
	TokenCache& tokenCache();

protected:
    // Listener
    void notifyMe(Notification *message);

	// MachServer::Timer
	void action();

    void clearReaders(Reader::Type type);

public: //@@@@
	void startSoftTokens();
	void loadSoftToken(Bundle *tokendBundle);

private:
	ServiceLevel mServiceLevel;	// level of service requested/determined

	std::string mCachePath;		// path to cache directory
	TokenCache *mTokenCache;	// cache object (lazy)

	typedef map<string, RefPointer<Reader> > ReaderMap;
	typedef set<RefPointer<Reader> > ReaderSet;
	ReaderMap mReaders;		// presently known PCSC Readers (aka slots)

	class Watcher : public Thread {
	public:
		Watcher(Server &server, TokenCache &tokenCache, ReaderMap& readers);

	protected:
		void threadAction();

	private:
		Server &mServer;
		TokenCache &mTokenCache;
		PCSC::Session mSession;		// PCSC client session
		ReaderMap& mReaders;
	};
};


#endif //_H_PCSCMONITOR
