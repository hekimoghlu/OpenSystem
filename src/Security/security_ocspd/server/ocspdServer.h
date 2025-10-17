/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 22, 2022.
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
#ifndef	_OCSPD_SERVER_H_
#define _OCSPD_SERVER_H_

#include <security_utilities/machserver.h>
#include <Security/cssmtype.h>
#include <security_ocspd/ocspd.h>						/* created by MIG */

#define MAX_OCSPD_THREADS		128

void ServerActivity();


class OcspdServer : public MachPlusPlus::MachServer
{
	NOCOPY(OcspdServer)
public:
	OcspdServer(const char *bootstrapName);
	~OcspdServer();

	Allocator		&alloc()	{ return mAlloc; }
	static OcspdServer &active() 
								{ return safer_cast<OcspdServer &>(MachServer::active()); }
	
protected:
    // implementation methods of MachServer
	boolean_t handle(mach_msg_header_t *in, mach_msg_header_t *out);
	
	/* 
	 * Timer subclass to handle periodic flushes of DB caches.
	 */
	class OcspdTimer : public MachServer::Timer
	{
		NOCOPY(OcspdTimer)
	public:
		/* Timer(false) --> !longTerm --> avoid spawning a thread for this */
		OcspdTimer(OcspdServer &server) : Timer(true), mServer(server) {}
		virtual ~OcspdTimer() {}
		virtual void action();
	private:
		OcspdServer &mServer;
	};

	/* we're not handling dead port notification for now */
private:
	Allocator		&mAlloc;
	OcspdTimer		mTimer;
};

/*
 * Given a CSSM_DATA which was allocated in our server's alloc space, 
 * pass referent data back to caller and schedule a dealloc after the RPC
 * completes with MachServer.
 */
extern void passDataToCaller(
	CSSM_DATA		&srcData,		// allocd in our server's alloc space
	Data			*outData,
	mach_msg_type_number_t *outDataCnt);

#endif	/* _OCSPD_SERVER_H_ */


