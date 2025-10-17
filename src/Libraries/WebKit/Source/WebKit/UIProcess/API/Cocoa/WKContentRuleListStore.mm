/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 10, 2024.
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
#import "WKContentRuleListInternal.h"
#import "WKContentRuleListStoreInternal.h"

#import "APIContentRuleListStore.h"
#import "NetworkCacheFileSystem.h"
#import "WKErrorInternal.h"
#import <WebCore/WebCoreObjCExtras.h>
#import <wtf/BlockPtr.h>
#import <wtf/CompletionHandler.h>
#import <wtf/cocoa/VectorCocoa.h>

#if ENABLE(CONTENT_EXTENSIONS)
static WKErrorCode toWKErrorCode(const std::error_code& error)
{
    ASSERT(error.category() == API::contentRuleListStoreErrorCategory());
    switch (static_cast<API::ContentRuleListStore::Error>(error.value())) {
    case API::ContentRuleListStore::Error::LookupFailed:
        return WKErrorContentRuleListStoreLookUpFailed;
    case API::ContentRuleListStore::Error::VersionMismatch:
        return WKErrorContentRuleListStoreVersionMismatch;
    case API::ContentRuleListStore::Error::CompileFailed:
        return WKErrorContentRuleListStoreCompileFailed;
    case API::ContentRuleListStore::Error::RemoveFailed:
        return WKErrorContentRuleListStoreRemoveFailed;
    }
    ASSERT_NOT_REACHED();
    return WKErrorUnknown;
}
#endif

@implementation WKContentRuleListStore

WK_OBJECT_DISABLE_DISABLE_KVC_IVAR_ACCESS;

- (void)dealloc
{
    if (WebCoreObjCScheduleDeallocateOnMainRunLoop(WKContentRuleListStore.class, self))
        return;

    _contentRuleListStore->~ContentRuleListStore();

    [super dealloc];
}

+ (instancetype)defaultStore
{
#if ENABLE(CONTENT_EXTENSIONS)
    return wrapper(API::ContentRuleListStore::defaultStoreSingleton());
#else
    return nil;
#endif
}

+ (instancetype)storeWithURL:(NSURL *)url
{
#if ENABLE(CONTENT_EXTENSIONS)
    return wrapper(API::ContentRuleListStore::storeWithPath(url.absoluteURL.path)).autorelease();
#else
    return nil;
#endif
}

- (void)compileContentRuleListForIdentifier:(NSString *)identifier encodedContentRuleList:(NSString *)encodedContentRuleList completionHandler:(void (^)(WKContentRuleList *, NSError *))completionHandler
{
#if ENABLE(CONTENT_EXTENSIONS)
    _contentRuleListStore->compileContentRuleList(identifier, encodedContentRuleList, [completionHandler = makeBlockPtr(completionHandler)](RefPtr<API::ContentRuleList> contentRuleList, std::error_code error) {
        if (error) {
            auto userInfo = @{ NSHelpAnchorErrorKey: [NSString stringWithFormat:@"Rule list compilation failed: %s", error.message().c_str()] };

            // error.value() could have a specific compiler error that is not equal to WKErrorContentRuleListStoreCompileFailed.
            // We want to use error.message, but here we want to only pass on CompileFailed with userInfo from the std::error_code.
            return completionHandler(nil, [NSError errorWithDomain:WKErrorDomain code:WKErrorContentRuleListStoreCompileFailed userInfo:userInfo]);
        }
        completionHandler(wrapper(*contentRuleList), nil);
    });
#endif
}

- (void)lookUpContentRuleListForIdentifier:(NSString *)identifier completionHandler:(void (^)(WKContentRuleList *, NSError *))completionHandler
{
#if ENABLE(CONTENT_EXTENSIONS)
    _contentRuleListStore->lookupContentRuleList(identifier, [completionHandler = makeBlockPtr(completionHandler)](RefPtr<API::ContentRuleList> contentRuleList, std::error_code error) {
        if (error) {
            auto userInfo = @{NSHelpAnchorErrorKey: [NSString stringWithFormat:@"Rule list lookup failed: %s", error.message().c_str()]};
            auto wkError = toWKErrorCode(error);
            ASSERT(wkError == WKErrorContentRuleListStoreLookUpFailed || wkError == WKErrorContentRuleListStoreVersionMismatch);
            return completionHandler(nil, [NSError errorWithDomain:WKErrorDomain code:wkError userInfo:userInfo]);
        }

        completionHandler(wrapper(*contentRuleList), nil);
    });
#endif
}

- (void)getAvailableContentRuleListIdentifiers:(void (^)(NSArray<NSString *>*))completionHandler
{
#if ENABLE(CONTENT_EXTENSIONS)
    _contentRuleListStore->getAvailableContentRuleListIdentifiers([completionHandler = makeBlockPtr(completionHandler)](Vector<String> identifiers) {
        completionHandler(createNSArray(identifiers).get());
    });
#endif
}

- (void)removeContentRuleListForIdentifier:(NSString *)identifier completionHandler:(void (^)(NSError *))completionHandler
{
#if ENABLE(CONTENT_EXTENSIONS)
    _contentRuleListStore->removeContentRuleList(identifier, [completionHandler = makeBlockPtr(completionHandler)](std::error_code error) {
        if (error) {
            auto userInfo = @{NSHelpAnchorErrorKey: [NSString stringWithFormat:@"Rule list removal failed: %s", error.message().c_str()]};
            ASSERT(toWKErrorCode(error) == WKErrorContentRuleListStoreRemoveFailed);
            return completionHandler([NSError errorWithDomain:WKErrorDomain code:WKErrorContentRuleListStoreRemoveFailed userInfo:userInfo]);
        }

        completionHandler(nil);
    });
#endif
}

#pragma mark WKObject protocol implementation

- (API::Object&)_apiObject
{
    return *_contentRuleListStore;
}

@end

@implementation WKContentRuleListStore (WKPrivate)

// For testing only.

- (void)_removeAllContentRuleLists
{
#if ENABLE(CONTENT_EXTENSIONS)
    _contentRuleListStore->synchronousRemoveAllContentRuleLists();
#endif
}

- (void)_invalidateContentRuleListVersionForIdentifier:(NSString *)identifier
{
#if ENABLE(CONTENT_EXTENSIONS)
    _contentRuleListStore->invalidateContentRuleListVersion(identifier);
#endif
}

- (void)_corruptContentRuleListHeaderForIdentifier:(NSString *)identifier usingCurrentVersion:(BOOL)usingCurrentVersion
{
#if ENABLE(CONTENT_EXTENSIONS)
    _contentRuleListStore->corruptContentRuleListHeader(identifier, usingCurrentVersion);
#endif
}

- (void)_corruptContentRuleListActionsMatchingEverythingForIdentifier:(NSString *)identifier
{
#if ENABLE(CONTENT_EXTENSIONS)
    _contentRuleListStore->corruptContentRuleListActionsMatchingEverything(identifier);
#endif
}

- (void)_invalidateContentRuleListHeaderForIdentifier:(NSString *)identifier
{
#if ENABLE(CONTENT_EXTENSIONS)
    _contentRuleListStore->invalidateContentRuleListHeader(identifier);
#endif
}

- (void)_getContentRuleListSourceForIdentifier:(NSString *)identifier completionHandler:(void (^)(NSString*))completionHandler
{
#if ENABLE(CONTENT_EXTENSIONS)
    auto handler = adoptNS([completionHandler copy]);
    _contentRuleListStore->getContentRuleListSource(identifier, [handler](String source) {
        auto rawHandler = (void (^)(NSString *))handler.get();
        if (source.isNull()) {
            // This should not be necessary since there are no nullability annotations
            // in this file or any other unified source combined herein.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnonnull"
            rawHandler(nil);
#pragma clang diagnostic pop
        } else
            rawHandler(source);
    });
#endif
}

+ (instancetype)defaultStoreWithLegacyFilename
{
#if ENABLE(CONTENT_EXTENSIONS)
    return wrapper(API::ContentRuleListStore::defaultStoreSingleton());
#else
    return nil;
#endif
}

+ (instancetype)storeWithURLAndLegacyFilename:(NSURL *)url
{
#if ENABLE(CONTENT_EXTENSIONS)
    return wrapper(API::ContentRuleListStore::storeWithPath(url.absoluteURL.path)).autorelease();
#else
    return nil;
#endif
}

- (void)removeContentExtensionForIdentifier:(NSString *)identifier completionHandler:(void (^)(NSError *))completionHandler
{
    [self removeContentRuleListForIdentifier:identifier completionHandler:completionHandler];
}

@end
