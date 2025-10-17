/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 19, 2023.
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
#if !__has_feature(objc_arc)
#error This file requires ARC. Add the "-fobjc-arc" compiler flag for this file.
#endif

#import "config.h"
#import "WKWebExtensionControllerInternal.h"

#import "WKWebExtensionContextInternal.h"
#import "WKWebExtensionControllerConfigurationInternal.h"
#import "WKWebExtensionDataRecordInternal.h"
#import "WKWebExtensionDataTypeInternal.h"
#import "WKWebExtensionInternal.h"
#import "WebExtensionContext.h"
#import "WebExtensionController.h"
#import "WebExtensionDataType.h"
#import <WebCore/EventRegion.h>
#import <wtf/BlockPtr.h>
#import <wtf/CompletionHandler.h>

@implementation WKWebExtensionController

#if ENABLE(WK_WEB_EXTENSIONS)

WK_OBJECT_DEALLOC_IMPL_ON_MAIN_THREAD(WKWebExtensionController, WebExtensionController, Ref { *_webExtensionController });

- (instancetype)init
{
    if (!(self = [super init]))
        return nil;

    API::Object::constructInWrapper<WebKit::WebExtensionController>(self, WebKit::WebExtensionControllerConfiguration::createDefault());

    return self;
}

- (instancetype)initWithConfiguration:(WKWebExtensionControllerConfiguration *)configuration
{
    NSParameterAssert([configuration isKindOfClass:WKWebExtensionControllerConfiguration.class]);

    if (!(self = [super init]))
        return nil;

    API::Object::constructInWrapper<WebKit::WebExtensionController>(self, configuration._protectedWebExtensionControllerConfiguration->copy());

    return self;
}

- (WKWebExtensionControllerConfiguration *)configuration
{
    return Ref { *_webExtensionController }->protectedConfiguration()->copy()->wrapper();
}

- (BOOL)loadExtensionContext:(WKWebExtensionContext *)extensionContext error:(NSError **)outError
{
    NSParameterAssert([extensionContext isKindOfClass:WKWebExtensionContext.class]);

    return Ref { *_webExtensionController }->load(Ref { extensionContext._webExtensionContext }, outError);
}

- (BOOL)unloadExtensionContext:(WKWebExtensionContext *)extensionContext error:(NSError **)outError
{
    NSParameterAssert([extensionContext isKindOfClass:WKWebExtensionContext.class]);

    return Ref { *_webExtensionController }->unload(Ref { extensionContext._webExtensionContext }, outError);
}

- (WKWebExtensionContext *)extensionContextForExtension:(WKWebExtension *)extension
{
    NSParameterAssert([extension isKindOfClass:WKWebExtension.class]);

    if (auto extensionContext = Ref { *_webExtensionController }->extensionContext(extension._protectedWebExtension))
        return extensionContext->wrapper();
    return nil;
}

- (WKWebExtensionContext *)extensionContextForURL:(NSURL *)url
{
    NSParameterAssert([url isKindOfClass:NSURL.class]);

    if (auto extensionContext = Ref { *_webExtensionController }->extensionContext(url))
        return extensionContext->wrapper();
    return nil;
}

template<typename T>
static inline NSSet *toAPI(const HashSet<Ref<T>>& inputSet)
{
    if (inputSet.isEmpty())
        return [NSSet set];

    NSMutableSet *result = [[NSMutableSet alloc] initWithCapacity:inputSet.size()];

    for (auto& entry : inputSet)
        [result addObject:entry->wrapper()];

    return [result copy];
}

- (NSSet<WKWebExtension *> *)extensions
{
    return toAPI(Ref { *_webExtensionController }->extensions());
}

- (NSSet<WKWebExtensionContext *> *)extensionContexts
{
    return toAPI(Ref { *_webExtensionController }->extensionContexts());
}

+ (NSSet<WKWebExtensionDataType> *)allExtensionDataTypes
{
    return [NSSet setWithObjects:WKWebExtensionDataTypeLocal, WKWebExtensionDataTypeSession, WKWebExtensionDataTypeSynchronized, nil];
}

- (void)fetchDataRecordsOfTypes:(NSSet<WKWebExtensionDataType> *)dataTypes completionHandler:(void (^)(NSArray<WKWebExtensionDataRecord *> *))completionHandler
{
    NSParameterAssert([dataTypes isKindOfClass:NSSet.class]);
    NSParameterAssert(completionHandler);

    Ref { *_webExtensionController }->getDataRecords(WebKit::toWebExtensionDataTypes(dataTypes), [completionHandler = makeBlockPtr(completionHandler)](Vector<Ref<WebKit::WebExtensionDataRecord>> records) {
        completionHandler(toAPI(records));
    });
}

- (void)fetchDataRecordOfTypes:(NSSet<WKWebExtensionDataType> *)dataTypes forExtensionContext:(WKWebExtensionContext *)extensionContext completionHandler:(void (^)(WKWebExtensionDataRecord *))completionHandler
{
    NSParameterAssert([dataTypes isKindOfClass:NSSet.class]);
    NSParameterAssert([extensionContext isKindOfClass:WKWebExtensionContext.class]);
    NSParameterAssert(completionHandler);

    Ref { *_webExtensionController }->getDataRecord(WebKit::toWebExtensionDataTypes(dataTypes), Ref { extensionContext._webExtensionContext }, [completionHandler = makeBlockPtr(completionHandler)](RefPtr<WebKit::WebExtensionDataRecord> record) {
        if (record)
            completionHandler(record->wrapper());
        else
            completionHandler(nil);
    });
}

- (void)removeDataOfTypes:(NSSet<WKWebExtensionDataType> *)dataTypes fromDataRecords:(NSArray<WKWebExtensionDataRecord *> *)dataRecords completionHandler:(void (^)())completionHandler
{
    NSParameterAssert([dataTypes isKindOfClass:NSSet.class]);
    NSParameterAssert([dataRecords isKindOfClass:NSArray.class]);
    NSParameterAssert(completionHandler);

    Ref { *_webExtensionController }->removeData(WebKit::toWebExtensionDataTypes(dataTypes), WebKit::toWebExtensionDataRecords(dataRecords), [completionHandler = makeBlockPtr(completionHandler)] {
        completionHandler();
    });
}

- (void)didOpenWindow:(id<WKWebExtensionWindow>)newWindow
{
    NSParameterAssert(newWindow != nil);

    for (auto& context : Ref { *_webExtensionController }->extensionContexts())
        [context->wrapper() didOpenWindow:newWindow];
}

- (void)didCloseWindow:(id<WKWebExtensionWindow>)closedWindow
{
    NSParameterAssert(closedWindow != nil);

    for (auto& context : Ref { *_webExtensionController }->extensionContexts())
        [context->wrapper() didCloseWindow:closedWindow];
}

- (void)didFocusWindow:(id<WKWebExtensionWindow>)focusedWindow
{
    for (auto& context : Ref { *_webExtensionController }->extensionContexts())
        [context->wrapper() didFocusWindow:focusedWindow];
}

- (void)didOpenTab:(id<WKWebExtensionTab>)newTab
{
    NSParameterAssert(newTab != nil);

    for (auto& context : Ref { *_webExtensionController }->extensionContexts())
        [context->wrapper() didOpenTab:newTab];
}

- (void)didCloseTab:(id<WKWebExtensionTab>)closedTab windowIsClosing:(BOOL)windowIsClosing
{
    NSParameterAssert(closedTab != nil);

    for (auto& context : Ref { *_webExtensionController }->extensionContexts())
        [context->wrapper() didCloseTab:closedTab windowIsClosing:windowIsClosing];
}

- (void)didActivateTab:(id<WKWebExtensionTab>)activatedTab previousActiveTab:(nullable id<WKWebExtensionTab>)previousTab
{
    NSParameterAssert(activatedTab != nil);

    for (auto& context : Ref { *_webExtensionController }->extensionContexts())
        [context->wrapper() didActivateTab:activatedTab previousActiveTab:previousTab];
}

- (void)didSelectTabs:(NSArray<id<WKWebExtensionTab>> *)selectedTabs
{
    NSParameterAssert([selectedTabs isKindOfClass:NSArray.class]);

    for (auto& context : Ref { *_webExtensionController }->extensionContexts())
        [context->wrapper() didSelectTabs:selectedTabs];
}

- (void)didDeselectTabs:(NSArray<id<WKWebExtensionTab>> *)deselectedTabs
{
    NSParameterAssert([deselectedTabs isKindOfClass:NSArray.class]);

    for (auto& context : Ref { *_webExtensionController }->extensionContexts())
        [context->wrapper() didDeselectTabs:deselectedTabs];
}

- (void)didMoveTab:(id<WKWebExtensionTab>)movedTab fromIndex:(NSUInteger)index inWindow:(id<WKWebExtensionWindow>)oldWindow
{
    NSParameterAssert(movedTab != nil);

    for (auto& context : Ref { *_webExtensionController }->extensionContexts())
        [context->wrapper() didMoveTab:movedTab fromIndex:index inWindow:oldWindow];
}

- (void)didReplaceTab:(id<WKWebExtensionTab>)oldTab withTab:(id<WKWebExtensionTab>)newTab
{
    NSParameterAssert(oldTab != nil);
    NSParameterAssert(newTab != nil);

    for (auto& context : Ref { *_webExtensionController }->extensionContexts())
        [context->wrapper() didReplaceTab:oldTab withTab:newTab];
}

- (void)didChangeTabProperties:(WKWebExtensionTabChangedProperties)properties forTab:(id<WKWebExtensionTab>)changedTab
{
    NSParameterAssert(changedTab != nil);

    for (auto& context : Ref { *_webExtensionController }->extensionContexts())
        [context->wrapper() didChangeTabProperties:properties forTab:changedTab];
}

- (BOOL)_inTestingMode
{
    return _webExtensionController->inTestingMode();
}

- (void)_setTestingMode:(BOOL)testingMode
{
    _webExtensionController->setTestingMode(testingMode);
}

#pragma mark WKObject protocol implementation

- (API::Object&)_apiObject
{
    return *_webExtensionController;
}

- (WebKit::WebExtensionController&)_webExtensionController
{
    return *_webExtensionController;
}

#else // ENABLE(WK_WEB_EXTENSIONS)

- (instancetype)init
{
    return nil;
}

- (instancetype)initWithConfiguration:(WKWebExtensionControllerConfiguration *)configuration
{
    return nil;
}

- (WKWebExtensionControllerConfiguration *)configuration
{
    return nil;
}

- (BOOL)loadExtensionContext:(WKWebExtensionContext *)extensionContext error:(NSError **)error
{
    if (error)
        *error = [NSError errorWithDomain:NSCocoaErrorDomain code:NSFeatureUnsupportedError userInfo:nil];
    return NO;
}

- (BOOL)unloadExtensionContext:(WKWebExtensionContext *)extensionContext error:(NSError **)error
{
    if (error)
        *error = [NSError errorWithDomain:NSCocoaErrorDomain code:NSFeatureUnsupportedError userInfo:nil];
    return NO;
}

- (WKWebExtensionContext *)extensionContextForExtension:(WKWebExtension *)extension
{
    return nil;
}

- (WKWebExtensionContext *)extensionContextForURL:(NSURL *)url
{
    return nil;
}

- (NSSet<WKWebExtension *> *)extensions
{
    return nil;
}

- (NSSet<WKWebExtensionContext *> *)extensionContexts
{
    return nil;
}

+ (NSSet<WKWebExtensionDataType> *)allExtensionDataTypes
{
    return nil;
}

- (void)fetchDataRecordsOfTypes:(NSSet<WKWebExtensionDataType> *)dataTypes completionHandler:(void (^)(NSArray<WKWebExtensionDataRecord *> *))completionHandler
{
    completionHandler(@[ ]);
}

- (void)fetchDataRecordOfTypes:(NSSet<WKWebExtensionDataType> *)dataTypes forExtensionContext:(WKWebExtensionContext *)extensionContext completionHandler:(void (^)(WKWebExtensionDataRecord *))completionHandler
{
    completionHandler(nil);
}

- (void)removeDataOfTypes:(NSSet<WKWebExtensionDataType> *)dataTypes fromDataRecords:(NSArray<WKWebExtensionDataRecord *> *)dataRecords completionHandler:(void (^)())completionHandler
{
    completionHandler();
}

- (void)didOpenWindow:(id<WKWebExtensionWindow>)newWindow
{
}

- (void)didCloseWindow:(id<WKWebExtensionWindow>)closedWindow
{
}

- (void)didFocusWindow:(id<WKWebExtensionWindow>)focusedWindow
{
}

- (void)didOpenTab:(id<WKWebExtensionTab>)newTab
{
}

- (void)didCloseTab:(id<WKWebExtensionTab>)closedTab windowIsClosing:(BOOL)windowIsClosing
{
}

- (void)didActivateTab:(id<WKWebExtensionTab>)activatedTab previousActiveTab:(nullable id<WKWebExtensionTab>)previousTab
{
}

- (void)didSelectTabs:(NSSet<id<WKWebExtensionTab>> *)selectedTabs
{
}

- (void)didDeselectTabs:(NSSet<id<WKWebExtensionTab>> *)deselectedTabs
{
}

- (void)didMoveTab:(id<WKWebExtensionTab>)movedTab fromIndex:(NSUInteger)index inWindow:(id<WKWebExtensionWindow>)oldWindow
{
}

- (void)didReplaceTab:(id<WKWebExtensionTab>)oldTab withTab:(id<WKWebExtensionTab>)newTab
{
}

- (void)didChangeTabProperties:(WKWebExtensionTabChangedProperties)properties forTab:(id<WKWebExtensionTab>)changedTab
{
}

- (BOOL)_inTestingMode
{
    return NO;
}

- (void)_setTestingMode:(BOOL)testingMode
{
}

#endif // ENABLE(WK_WEB_EXTENSIONS)

@end
