/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 27, 2022.
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
#import "config.h"
#import "_WKInspectorDebuggableInfoInternal.h"

#import <WebCore/WebCoreObjCExtras.h>

@implementation _WKInspectorDebuggableInfo

- (instancetype)init
{
    if (!(self = [super init]))
        return nil;

    API::Object::constructInWrapper<API::DebuggableInfo>(self);

    return self;
}

- (_WKInspectorDebuggableType)debuggableType
{
    return toWKInspectorDebuggableType(_debuggableInfo->debuggableType());
}

- (void)setDebuggableType:(_WKInspectorDebuggableType)debuggableType
{
    _debuggableInfo->setDebuggableType(fromWKInspectorDebuggableType(debuggableType));
}

- (NSString *)targetPlatformName
{
    return _debuggableInfo->targetPlatformName();
}

- (void)setTargetPlatformName:(NSString *)targetPlatformName
{
    _debuggableInfo->setTargetPlatformName(targetPlatformName);
}

- (NSString *)targetBuildVersion
{
    return _debuggableInfo->targetBuildVersion();
}

- (void)setTargetBuildVersion:(NSString *)targetBuildVersion
{
    _debuggableInfo->setTargetBuildVersion(targetBuildVersion);
}

- (NSString *)targetProductVersion
{
    return _debuggableInfo->targetProductVersion();
}

- (void)setTargetProductVersion:(NSString *)targetProductVersion
{
    _debuggableInfo->setTargetProductVersion(targetProductVersion);
}

- (BOOL)targetIsSimulator
{
    return _debuggableInfo->targetIsSimulator();
}

- (void)setTargetIsSimulator:(BOOL)targetIsSimulator
{
    _debuggableInfo->setTargetIsSimulator(targetIsSimulator);
}

- (void)dealloc
{
    if (WebCoreObjCScheduleDeallocateOnMainRunLoop(_WKInspectorDebuggableInfo.class, self))
        return;

    _debuggableInfo->~DebuggableInfo();

    [super dealloc];
}

- (id)copyWithZone:(NSZone *)zone
{
    _WKInspectorDebuggableInfo *debuggableInfo = [(_WKInspectorDebuggableInfo *)[[self class] allocWithZone:zone] init];

    debuggableInfo.debuggableType = self.debuggableType;
    debuggableInfo.targetPlatformName = self.targetPlatformName;
    debuggableInfo.targetBuildVersion = self.targetBuildVersion;
    debuggableInfo.targetProductVersion = self.targetProductVersion;
    debuggableInfo.targetIsSimulator = self.targetIsSimulator;

    return debuggableInfo;
}

#pragma mark WKObject protocol implementation

- (API::Object&)_apiObject
{
    return *_debuggableInfo;
}

@end
