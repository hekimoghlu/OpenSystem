/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 11, 2022.
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
#import "WKUserScriptInternal.h"

#import "WKContentWorldInternal.h"
#import "_WKUserContentWorldInternal.h"
#import <WebCore/WebCoreObjCExtras.h>
#import <wtf/cocoa/VectorCocoa.h>

@implementation WKUserScript

WK_OBJECT_DISABLE_DISABLE_KVC_IVAR_ACCESS;

- (instancetype)initWithSource:(NSString *)source injectionTime:(WKUserScriptInjectionTime)injectionTime forMainFrameOnly:(BOOL)forMainFrameOnly
{
    return [self initWithSource:source injectionTime:injectionTime forMainFrameOnly:forMainFrameOnly inContentWorld:WKContentWorld.pageWorld];
}

- (instancetype)initWithSource:(NSString *)source injectionTime:(WKUserScriptInjectionTime)injectionTime forMainFrameOnly:(BOOL)forMainFrameOnly inContentWorld:(WKContentWorld *)contentWorld
{
    if (!(self = [super init]))
        return nil;

    API::Object::constructInWrapper<API::UserScript>(self, WebCore::UserScript { source, { }, { }, { }, API::toWebCoreUserScriptInjectionTime(injectionTime), forMainFrameOnly ? WebCore::UserContentInjectedFrames::InjectInTopFrameOnly : WebCore::UserContentInjectedFrames::InjectInAllFrames }, *contentWorld->_contentWorld);

    return self;
}

- (void)dealloc
{
    if (WebCoreObjCScheduleDeallocateOnMainRunLoop(WKUserScript.class, self))
        return;

    _userScript->~UserScript();

    [super dealloc];
}

- (NSString *)source
{
    return _userScript->userScript().source();
}

- (WKUserScriptInjectionTime)injectionTime
{
    return API::toWKUserScriptInjectionTime(_userScript->userScript().injectionTime());
}

- (BOOL)isForMainFrameOnly
{
    return _userScript->userScript().injectedFrames() == WebCore::UserContentInjectedFrames::InjectInTopFrameOnly;
}

- (id)copyWithZone:(NSZone *)zone
{
    return [self retain];
}

#pragma mark WKObject protocol implementation

- (API::Object&)_apiObject
{
    return *_userScript;
}

@end

@implementation WKUserScript (WKPrivate)

ALLOW_DEPRECATED_DECLARATIONS_BEGIN
- (_WKUserContentWorld *)_userContentWorld
{
    return adoptNS([[_WKUserContentWorld alloc] _initWithContentWorld:wrapper(_userScript->contentWorld())]).autorelease();
}
ALLOW_DEPRECATED_DECLARATIONS_END

- (instancetype)_initWithSource:(NSString *)source injectionTime:(WKUserScriptInjectionTime)injectionTime forMainFrameOnly:(BOOL)forMainFrameOnly includeMatchPatternStrings:(NSArray<NSString *> *)includeMatchPatternStrings excludeMatchPatternStrings:(NSArray<NSString *> *)excludeMatchPatternStrings associatedURL:(NSURL *)associatedURL contentWorld:(WKContentWorld *)contentWorld deferRunningUntilNotification:(BOOL)deferRunningUntilNotification
{
    // deferRunningUntilNotification is no longer supported.
    NSParameterAssert(!deferRunningUntilNotification);

    return [self _initWithSource:source injectionTime:injectionTime forMainFrameOnly:forMainFrameOnly includeMatchPatternStrings:includeMatchPatternStrings excludeMatchPatternStrings:excludeMatchPatternStrings associatedURL:associatedURL contentWorld:contentWorld];
}

- (instancetype)_initWithSource:(NSString *)source injectionTime:(WKUserScriptInjectionTime)injectionTime forMainFrameOnly:(BOOL)forMainFrameOnly includeMatchPatternStrings:(NSArray<NSString *> *)includeMatchPatternStrings excludeMatchPatternStrings:(NSArray<NSString *> *)excludeMatchPatternStrings associatedURL:(NSURL *)associatedURL contentWorld:(WKContentWorld *)contentWorld
{
    if (!(self = [super init]))
        return nil;

    API::Object::constructInWrapper<API::UserScript>(self, WebCore::UserScript { source, associatedURL, makeVector<String>(includeMatchPatternStrings), makeVector<String>(excludeMatchPatternStrings), API::toWebCoreUserScriptInjectionTime(injectionTime), forMainFrameOnly ? WebCore::UserContentInjectedFrames::InjectInTopFrameOnly : WebCore::UserContentInjectedFrames::InjectInAllFrames }, contentWorld ? *contentWorld->_contentWorld : API::ContentWorld::pageContentWorldSingleton());

    return self;
}

- (WKContentWorld *)_contentWorld
{
    return wrapper(_userScript->contentWorld());
}

@end
