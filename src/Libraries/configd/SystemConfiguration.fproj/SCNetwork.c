/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 24, 2022.
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
 * Modification History
 *
 * January 19, 2003		Allan Nathanson <ajn@apple.com>
 * - add advanced reachability APIs
 *
 * June 10, 2001		Allan Nathanson <ajn@apple.com>
 * - updated to use service-based "State:" information
 *
 * June 1, 2001			Allan Nathanson <ajn@apple.com>
 * - public API conversion
 *
 * January 30, 2001		Allan Nathanson <ajn@apple.com>
 * - initial revision
 */

#include <sys/types.h>
#include <netinet/in.h>
#include <unistd.h>
#include <sys/socket.h>

#include <SystemConfiguration/SystemConfiguration.h>
#include <SystemConfiguration/SCPrivate.h>

Boolean
SCNetworkCheckReachabilityByAddress(const struct sockaddr	*address,
				    socklen_t			addrlen,
				    SCNetworkConnectionFlags	*flags)
{
	SCNetworkReachabilityRef		networkAddress;
	Boolean			ok;
	struct sockaddr_storage	ss;

	if (!address ||
	    (addrlen == 0) ||
	    (addrlen > (int)sizeof(struct sockaddr_storage))) {
		_SCErrorSet(kSCStatusInvalidArgument);
		return FALSE;
	}

	memset(&ss, 0, sizeof(ss));
	memcpy(&ss, address, addrlen);
	ss.ss_len = addrlen;

	networkAddress = SCNetworkReachabilityCreateWithAddress(NULL, (struct sockaddr *)&ss);
	if (networkAddress == NULL) {
		return FALSE;
	}

	ok = SCNetworkReachabilityGetFlags(networkAddress, flags);
	CFRelease(networkAddress);
	return ok;
}


Boolean
SCNetworkCheckReachabilityByName(const char			*nodename,
				 SCNetworkConnectionFlags	*flags)
{
	SCNetworkReachabilityRef	networkAddress;
	Boolean				ok;

	if (!nodename) {
		_SCErrorSet(kSCStatusInvalidArgument);
		return FALSE;
	}

	networkAddress = SCNetworkReachabilityCreateWithName(NULL, nodename);
	if (networkAddress == NULL) {
		return FALSE;
	}

	ok = SCNetworkReachabilityGetFlags(networkAddress, flags);
	CFRelease(networkAddress);
	return ok;
}
