/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 10, 2022.
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
#ifndef _H_TOKEND_SERVER
#define _H_TOKEND_SERVER


//
// server - master server loop for tokend
//
#include "SecTokend.h"
#include <security_utilities/logging.h>
#include <security_utilities/pcsc++.h>
#include <security_utilities/machserver.h>
#include <security_utilities/alloc.h>

namespace Security {
namespace Tokend {


//
// The server class that drives this tokend
//
    class Server : public MachPlusPlus::MachServer, public SecTokendCallbacks {
public:
	int operator() (int argc, const char *argv[], SecTokendCallbackFlags flags);

	const char *readerName() const { return mReaderName; }
	const PCSC::ReaderState &startupReaderState() const { return mStartupReaderState; }
	
	const char *tokenUid() const { return mTokenUid.c_str(); }
	void tokenUid(const char *uid) { mTokenUid = uid; }

	SecTokendCallbacks &callbacks() { return static_cast<SecTokendCallbacks &>(*this); }
	
	void termination(uint32 reason, uint32 options) __attribute__((noreturn));
	
protected:
	boolean_t handle(mach_msg_header_t *in, mach_msg_header_t *out);
	
private:
	const char *mReaderName;
	PCSC::ReaderState mStartupReaderState;
	std::string mTokenUid;
};


//
// The server singleton
//
extern Server *server;


}	// namespace Tokend
}	// namespace Security

#endif //_H_TOKEND_SERVER
