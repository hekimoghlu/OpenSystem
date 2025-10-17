/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 27, 2022.
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
// tdclient - Security tokend client interface library
//
#include "tdtransit.h"
#include <security_utilities/debugging.h>

using MachPlusPlus::check;
using MachPlusPlus::Bootstrap;


namespace Security {
namespace Tokend {


//
// Construct a client session
//
ClientSession::ClientSession(Allocator &std, Allocator &rtn)
	: ClientCommon(std, rtn)
{
}


//
// Destroy a session
//
ClientSession::~ClientSession()
{ }


//
// The default fault() notifier does nothing
//
void ClientSession::fault()
{
}


//
// Administrativa
//
void ClientSession::servicePort(Port p)
{
	// record service port
	assert(!mServicePort);	// no overwrite
	mServicePort = p;
	
	// come back if the service port dies (usually a tokend crash)
	mServicePort.requestNotify(mReplyPort);
}


} // end namespace Tokend
} // end namespace Security
