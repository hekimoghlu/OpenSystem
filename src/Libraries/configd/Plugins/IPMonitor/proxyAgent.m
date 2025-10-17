/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 7, 2025.
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

@interface ProxyAgent:ConfigAgent

@property (nonatomic) AgentType internalAgentType;
@property (nonatomic) NSString *internalAgentName;
@property (nonatomic) AgentSubType internalAgentSubType;

@end

@implementation ProxyAgent

@synthesize agentUUID;
@synthesize agentDescription;

+ (NSString *)agentType
{
	return @kConfigAgentTypeProxy;
}

- (instancetype)initWithParameters:(NSDictionary *)parameters
{
	self = [super initWithParameters:parameters];
	if (self) {
		NSString *intf = [parameters valueForKey:@kEntityName];
		NSNumber *subType = [parameters valueForKey:@kAgentSubType];
		NSString *type = [[self class] agentType];

		_internalAgentName = [NSString stringWithFormat:@"%@-%@", type, intf];
		_internalAgentSubType = [subType unsignedIntegerValue];
		_internalAgentType = kAgentTypeProxy;

		agentDescription = _internalAgentName;
		agentUUID = [super createUUIDForName:agentDescription];
		if (agentUUID == nil) {
			agentUUID = [NSUUID UUID];
		}
	}

	return self;
}

- (AgentType)getAgentType
{
	return _internalAgentType;
}

- (NSString *)getAgentName
{
	return _internalAgentName;
}

- (AgentSubType)getAgentSubType
{
	return _internalAgentSubType;
}

- (NSUUID *)getAgentUUID
{
	return agentUUID;
}

@end
