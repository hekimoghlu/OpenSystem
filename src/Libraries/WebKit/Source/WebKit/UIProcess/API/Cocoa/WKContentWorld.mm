/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 18, 2024.
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
#import "WKContentWorldInternal.h"

#import "_WKUserContentWorldInternal.h"
#import <WebCore/WebCoreObjCExtras.h>

static void checkContentWorldOptions(API::ContentWorld& world, _WKContentWorldConfiguration *configuration)
{
    if (world.allowAccessToClosedShadowRoots() != (configuration && configuration.allowAccessToClosedShadowRoots))
        [NSException raise:NSInternalInconsistencyException format:@"The value of allowAccessToClosedShadowRoots does not match the existing world"];
    if (world.allowAutofill() != (configuration && configuration.allowAutofill))
        [NSException raise:NSInternalInconsistencyException format:@"The value of allowAutofill does not match the existing world"];
    if (world.allowElementUserInfo() != (configuration && configuration.allowElementUserInfo))
        [NSException raise:NSInternalInconsistencyException format:@"The value of allowElementUserInfo does not match the existing world"];
    if (world.disableLegacyBuiltinOverrides() != (configuration && configuration.disableLegacyBuiltinOverrides))
        [NSException raise:NSInternalInconsistencyException format:@"The value of disableLegacyBuiltinOverrides does not match the existing world"];
}

@implementation WKContentWorld

WK_OBJECT_DISABLE_DISABLE_KVC_IVAR_ACCESS;

+ (WKContentWorld *)pageWorld
{
    return wrapper(API::ContentWorld::pageContentWorldSingleton());
}

+ (WKContentWorld *)defaultClientWorld
{
    return wrapper(API::ContentWorld::defaultClientWorldSingleton());
}

+ (WKContentWorld *)worldWithName:(NSString *)name
{
    Ref world = API::ContentWorld::sharedWorldWithName(name);
    checkContentWorldOptions(world, nil);
    return wrapper(WTFMove(world)).autorelease();
}

- (void)dealloc
{
    if (WebCoreObjCScheduleDeallocateOnMainRunLoop(WKContentWorld.class, self))
        return;

    _contentWorld->~ContentWorld();

    [super dealloc];
}

- (NSString *)name
{
    if (_contentWorld->name().isNull())
        return nil;

    return _contentWorld->name();
}

#pragma mark WKObject protocol implementation

- (API::Object&)_apiObject
{
    return *_contentWorld;
}

@end

@implementation WKContentWorld (WKPrivate)

ALLOW_DEPRECATED_DECLARATIONS_BEGIN
- (_WKUserContentWorld *)_userContentWorld
{
    return adoptNS([[_WKUserContentWorld alloc] _initWithContentWorld:self]).autorelease();
}
ALLOW_DEPRECATED_DECLARATIONS_END

+ (WKContentWorld *)_worldWithConfiguration:(_WKContentWorldConfiguration *)configuration
{
    OptionSet<WebKit::ContentWorldOption> optionSet;
    if (configuration.allowAccessToClosedShadowRoots)
        optionSet.add(WebKit::ContentWorldOption::AllowAccessToClosedShadowRoots);
    if (configuration.allowAutofill)
        optionSet.add(WebKit::ContentWorldOption::AllowAutofill);
    if (configuration.allowElementUserInfo)
        optionSet.add(WebKit::ContentWorldOption::AllowElementUserInfo);
    if (configuration.disableLegacyBuiltinOverrides)
        optionSet.add(WebKit::ContentWorldOption::DisableLegacyBuiltinOverrides);
    Ref world = API::ContentWorld::sharedWorldWithName(configuration.name, optionSet);
    checkContentWorldOptions(world, configuration);
    return wrapper(WTFMove(world)).autorelease();
}

@end
