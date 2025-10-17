/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 15, 2025.
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
#ifndef _SCNETWORKREACHABILITYINTERNAL_H
#define _SCNETWORKREACHABILITYINTERNAL_H

#include <os/availability.h>
#include <TargetConditionals.h>
#include <sys/cdefs.h>
#include <CoreFoundation/CoreFoundation.h>
#include <CoreFoundation/CFRuntime.h>
#include <SystemConfiguration/SystemConfiguration.h>
#include <SystemConfiguration/SCPrivate.h>
#include <dispatch/dispatch.h>

#include <netdb.h>
#include <sys/socket.h>
#include <net/if.h>
#include <xpc/xpc.h>

#include <nw/private.h>

#pragma mark -
#pragma mark SCNetworkReachability

#define kSCNetworkReachabilityFlagsMask			0x00ffffff	// top 8-bits reserved for implementation


typedef	enum {
	NO	= 0,
	YES,
	UNKNOWN
} lazyBoolean;

typedef enum {
	ReachabilityRankNone			= 0,
	ReachabilityRankConnectionRequired	= 1,
	ReachabilityRankReachable		= 2
} ReachabilityRankType;

typedef enum {
	// by-address SCNetworkReachability targets
	reachabilityTypeAddress,
	reachabilityTypeAddressPair,
	// by-name SCNetworkReachability targets
	reachabilityTypeName,
	reachabilityTypePTR
} ReachabilityAddressType;

#define isReachabilityTypeAddress(type)		(type < reachabilityTypeName)
#define isReachabilityTypeName(type)		(type >= reachabilityTypeName)

typedef struct {

	/* base CFType information */
	CFRuntimeBase			cfBase;

	/* lock */
	pthread_mutex_t			lock;

	/* address type */
	ReachabilityAddressType		type;

	/* target host name */
	nw_endpoint_t			hostnameEndpoint;

	/* local & remote addresses */
	nw_endpoint_t			localAddressEndpoint;
	nw_endpoint_t			remoteAddressEndpoint;

	/* run loop source, callout, context, rl scheduling info */
	Boolean				scheduled;
	Boolean				sentFirstUpdate;
	CFRunLoopSourceRef		rls;
	SCNetworkReachabilityCallBack	rlsFunction;
	SCNetworkReachabilityContext	rlsContext;
	CFMutableArrayRef		rlList;

	dispatch_queue_t		dispatchQueue;		// SCNetworkReachabilitySetDispatchQueue

	Boolean				resolverBypass;		// set this flag to bypass resolving the name

	/* logging */
	char				log_prefix[32];

	nw_parameters_t			parameters;
	nw_path_evaluator_t		pathEvaluator;
	nw_path_t			lastPath;
	nw_parameters_t			lastPathParameters;
	nw_resolver_t			resolver;
	nw_resolver_status_t		lastResolverStatus;
	nw_array_t			lastResolvedEndpoints;
	Boolean				lastResolvedEndpointHasFlags;
	SCNetworkReachabilityFlags	lastResolvedEndpointFlags;
	uint				lastResolvedEndpointInterfaceIndex;

} SCNetworkReachabilityPrivate, *SCNetworkReachabilityPrivateRef;


// ------------------------------------------------------------


__BEGIN_DECLS

static __inline__ ReachabilityRankType
__SCNetworkReachabilityRank(SCNetworkReachabilityFlags flags)
{
	ReachabilityRankType	rank = ReachabilityRankNone;

	if ((flags & kSCNetworkReachabilityFlagsReachable) != 0) {
		rank = ReachabilityRankReachable;
		if ((flags & kSCNetworkReachabilityFlagsConnectionRequired) != 0) {
			rank = ReachabilityRankConnectionRequired;
		}
	}
	return rank;
}


__END_DECLS

#endif	// _SCNETWORKREACHABILITYINTERNAL_H
