/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 11, 2024.
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
#ifndef CONFIG_AGENT_H
#define CONFIG_AGENT_H

#include <net/if.h>
#include <net/network_agent.h>
#include <net/necp.h>
#include <dnsinfo.h>
#include <sys/ioctl.h>
#include <network_information.h>
#include <notify.h>
#include <sys/kern_control.h>
#include <sys/sys_domain.h>
#include <SystemConfiguration/SystemConfiguration.h>
#include <SystemConfiguration/SCPrivate.h>

#import <Foundation/Foundation.h>
#import <Network/Network_Private.h>
#import <NetworkExtension/NEPolicySession.h>
#import <CommonCrypto/CommonDigest.h>

#import "configAgentDefines.h"

typedef NS_ENUM(NSUInteger, AgentType) {
	kAgentTypeUnknown = 0,
	kAgentTypeProxy,
	kAgentTypeDNS
};

typedef NS_ENUM(NSUInteger, AgentSubType) {
	kAgentSubTypeUnknown = 0,
	kAgentSubTypeScoped,
	kAgentSubTypeSupplemental,
	kAgentSubTypeDefault,
	kAgentSubTypeMulticast,
	kAgentSubTypePrivate,
	kAgentSubTypeServiceSpecific,
	kAgentSubTypeGlobal,
};

os_log_t	__log_IPMonitor(void);

/* Parameters */
#define kEntityName	"EntityName"
#define kAgentSubType	"AgentSubType"

@interface ConfigAgent : NSObject <NWNetworkAgent>

@property NEPolicySession *preferredPolicySession;

- (instancetype)initWithParameters:(NSDictionary *)parameters;
- (void)addAgentRegistrationObject:(NWNetworkAgentRegistration *)regObject;
- (NWNetworkAgentRegistration *)getRegistrationObject;
- (NSString *)getAssociatedEntity;
- (NSString *)getAgentName;
- (NSData *)getAgentData;
- (AgentType)getAgentType;
- (AgentSubType)getAgentSubType;
- (NSUUID *)getAgentUUID;
- (void)setStartHandler:(void (^)(void))startHandler;
- (BOOL)startAgentWithOptions:(NSDictionary *)options;
- (void)updateAgentData:(NSData *)data;
- (BOOL)shouldUpdateAgent;
- (id)getAgentMapping;
- (void)setAgentMapping:(id)agent;

- (NSUUID *)createUUIDForName:(NSString *)agentName;

@end

#endif /* CONFIG_AGENT_H */
