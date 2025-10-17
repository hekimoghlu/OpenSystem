/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 8, 2022.
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
#import "WebScriptWorld.h"

#import "WebScriptWorldInternal.h"
#import <WebCore/JSDOMBinding.h>
#import <WebCore/ScriptController.h>
#import <JavaScriptCore/APICast.h>
#import <JavaScriptCore/JSContextInternal.h>
#import <wtf/RefPtr.h>

@interface WebScriptWorldPrivate : NSObject {
@public
    RefPtr<WebCore::DOMWrapperWorld> world;
}
@end

@implementation WebScriptWorldPrivate
@end

using WorldMap = HashMap<SingleThreadWeakRef<WebCore::DOMWrapperWorld>, WebScriptWorld*>;
static WorldMap& allWorlds()
{
    static WorldMap& map = *new WorldMap;
    return map;
}

@implementation WebScriptWorld

- (id)initWithWorld:(Ref<WebCore::DOMWrapperWorld>&&)world
{
    self = [super init];
    if (!self)
        return nil;

    _private = [[WebScriptWorldPrivate alloc] init];
    _private->world = WTFMove(world);

    ASSERT_ARG(world, !allWorlds().contains(*_private->world));
    allWorlds().add(*_private->world, self);

    return self;
}

- (id)init
{
    return [self initWithWorld:WebCore::ScriptController::createWorld("WebScriptWorld"_s, WebCore::ScriptController::WorldType::User)];
}

- (void)unregisterWorld
{
    _private->world->clearWrappers();
}

- (void)dealloc
{
    ASSERT(allWorlds().contains(*_private->world));
    allWorlds().remove(*_private->world);

    [_private release];
    _private = nil;
    [super dealloc];
}

+ (WebScriptWorld *)standardWorld
{
    static WebScriptWorld *world = [[WebScriptWorld alloc] initWithWorld:WebCore::mainThreadNormalWorld()];
    return world;
}

+ (WebScriptWorld *)world
{
    return adoptNS([[self alloc] init]).autorelease();
}

+ (WebScriptWorld *)scriptWorldForGlobalContext:(JSGlobalContextRef)context
{
    return [self findOrCreateWorld:WebCore::currentWorld(*toJS(context))];
}

#if JSC_OBJC_API_ENABLED
+ (WebScriptWorld *)scriptWorldForJavaScriptContext:(JSContext *)context
{
    return [self scriptWorldForGlobalContext:[context JSGlobalContextRef]];
}
#endif

@end

@implementation WebScriptWorld (WebInternal)

WebCore::DOMWrapperWorld* core(WebScriptWorld *world)
{
    return world ? world->_private->world.get() : 0;
}

+ (WebScriptWorld *)findOrCreateWorld:(WebCore::DOMWrapperWorld&)world
{
    if (&world == &WebCore::mainThreadNormalWorld())
        return [self standardWorld];

    if (WebScriptWorld *existingWorld = allWorlds().get(world))
        return existingWorld;

    return adoptNS([[self alloc] initWithWorld:world]).autorelease();
}

@end
