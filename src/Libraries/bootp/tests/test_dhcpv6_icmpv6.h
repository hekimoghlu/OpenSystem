/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 5, 2023.
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
#ifndef test_dhcpv6_icmpv6_h
#define test_dhcpv6_icmpv6_h

#include <sys/types.h>
#include <sys/socket.h>
#import <netinet/in.h>
#import <netinet6/in6.h>
#import <netinet6/in6_var.h>
#import <netinet/icmp6.h>
#import <netinet/ip6.h>

#import "DHCPv6.h"
#import "DHCPv6Options.h"
#import "interfaces.h"
#import "IPv6Socket.h"

#import "test_utils.h"

#define DHCPV6_SERVER_QUEUE_LABEL "DHCPv6 Server Queue"
#define DHCPV6PD_SERVICE_QUEUE "DHCPv6 PD Service Queue"

#define ERR -1
#define NOERR 0
#define NONNULL_OR_ERROUT(ptr, str, ...) 	\
do {					\
if ((ptr) == NULL) {			\
NSLog(@str "\n", ## __VA_ARGS__);	\
return ERR;				\
}					\
} while (0)
#define NOERR_OR_ERROUT(err, str, ...) 		\
do {					\
if ((err) != 0) {			\
NSLog(@str "\n", ## __VA_ARGS__);	\
return ERR;				\
}					\
} while (0)

// fc00::10:0:10:1
#define DHCPV6_SERVER_ULA_IN6ADDR 				\
{{{ 0xfc, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,		\
0x00, 0x10, 0x00, 0x00, 0x00, 0x10, 0x00, 0x01 }}}

// fc00::10:0:11:1
#define DHCPV6_CLIENT_ULA_IN6ADDR 				\
{{{ 0xfc, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,		\
0x00, 0x10, 0x00, 0x00, 0x00, 0x11, 0x00, 0x01 }}}

#define FETH0_LINKADDR_BYTES { 0x66, 0x65, 0x74, 0x68, 0x00, 0x00 }
#define FETH0_LINKADDR_LEN 6
#define FETH1_LINKADDR_BYTES { 0x66, 0x65, 0x74, 0x68, 0x00, 0x01 }
#define FETH1_LINKADDR_LEN 6

typedef enum {
	kDHCPServerFailureModeNone = 0,
	kDHCPServerFailureModeNotOnLink = 1,
	kDHCPServerFailureModeNoPrefixAvail = 2
} DHCPServerFailureMode;

@interface DHCPv6Server : NSObject
@property int socket;
@property dispatch_source_t socketListener;
@property dispatch_queue_t queue;
@property int interfaceIndex;
@property NSData * duid;
@property BOOL clientConfigured;
@property DHCPServerFailureMode failureMode;
@property NSDate * timeOfRequest;
@property NSTimeInterval timeBetweenSubsequentRequests1;
@property NSTimeInterval timeBetweenSubsequentRequests2;
@property dispatch_semaphore_t exponentialBackoffSem;

- (instancetype)initWithFailureMode:(DHCPServerFailureMode)failureMode
		       andInterface:(NSString *)ifname;
- (void)disconnect;
@end

#if 0
@interface ICMPv6Router: NSObject
@property int socket;
@property dispatch_source_t socketListener;
@property dispatch_queue_t queue;
@property int interfaceIndex;
@property NSData * duid;
@property BOOL clientConfigured;
@end
#endif

#endif /* test_dhcpv6_icmpv6_h */
