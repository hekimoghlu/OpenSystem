/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 12, 2022.
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
#import "configAgent.h"

@interface ConfigAgent()

@property (nonatomic) NWNetworkAgentRegistration * internalRegistrationObject;
@property (nonatomic) NSString *internalAssociatedEntity;
@property (nonatomic, copy) NSData *internalAgentData;
@property (nonatomic) BOOL internalShouldUpdateAgent;
@property (strong) void (^internalStartHandler)(void);
@property (nonatomic) id internalAgentMapping;

@end

@implementation ConfigAgent

@synthesize agentUUID;
@synthesize agentDescription;
@synthesize active;
@synthesize kernelActivated;
@synthesize userActivated;
@synthesize voluntary;
@synthesize specificUseOnly;

+ (NSString *)agentDomain
{
	return @kConfigAgentDomain;
}

+ (NSString *)agentType
{
	return @kConfigAgentTypeGeneric;
}

+ (instancetype)agentFromData:(NSData *)data
{
#pragma unused(data)
	return nil;
}

- (instancetype)initWithParameters:(NSDictionary *)parameters
{
	self = [super init];
	if (self) {
		NSString *intf = [parameters valueForKey:@kEntityName];

		_internalRegistrationObject = nil;
		_internalAssociatedEntity = [intf copy];
		_internalAgentData = nil;
		_internalShouldUpdateAgent = YES;
		_internalAgentMapping = nil;

		active = YES;
		kernelActivated = NO;
		userActivated = YES;
		voluntary = NO;
	}

	return self;
}

- (void)addAgentRegistrationObject:(NWNetworkAgentRegistration *)regObject
{
	_internalRegistrationObject = regObject;
}

- (AgentType)getAgentType
{
	return kAgentTypeUnknown;
}

- (NSUUID *)getAgentUUID
{
	return nil;
}

- (NSString *)getAgentName
{
	return @kConfigAgentTypeGeneric;
}

- (AgentSubType)getAgentSubType
{
	return kAgentSubTypeUnknown;
}

- (NWNetworkAgentRegistration *)getRegistrationObject
{
	return _internalRegistrationObject;
}

- (NSString *)getAssociatedEntity
{
	return _internalAssociatedEntity;
}

- (NSData *)getAgentData
{
	return _internalAgentData;
}

- (NSData *)copyAgentData
{
	return _internalAgentData;
}

- (void)setAgentMapping:(id)agent
{
	_internalAgentMapping = agent;
}

- (id)getAgentMapping
{
	return _internalAgentMapping;
}

- (void)setStartHandler:(void (^)(void))startHandler
{
	if (startHandler != nil) {
		self.internalStartHandler = startHandler;
	}
}

- (BOOL)startAgentWithOptions:(NSDictionary *)options
{
#pragma unused(options)
	BOOL ok = NO;
	if (!self.active) {
		self.active = YES;
		ok = [self.internalRegistrationObject updateNetworkAgent:self];
	}

	return ok;
}

- (void)updateAgentData:(NSData *)data
{
	if ([data isEqual:_internalAgentData]) {
		_internalShouldUpdateAgent = NO;
		return;
	}

	_internalAgentData = [data copy];
	_internalShouldUpdateAgent = YES;
}

- (BOOL)shouldUpdateAgent
{
	return _internalShouldUpdateAgent;
}

- (NSUUID *)createUUIDForName:(NSString *)agentName
{
	/* We would like to have same UUIDs for an interface/domain. So here is a way to fix this,
	 without maintaining any state in configd.

	 - We know that every agent has a unique name.
	 - Use that name to calculate an SHA256 hash.
	 - create a NSUUID from the first use the first 16 bytes of the hash.

	 - So for a name, we would always have the same UUID.

	 */
	unsigned char hashValue[CC_SHA256_DIGEST_LENGTH];
	const char *strForHash = [agentName UTF8String];
	CC_SHA256(strForHash, (CC_LONG)strlen(strForHash), hashValue);

	return [[NSUUID alloc] initWithUUIDBytes:hashValue];
}

@end
