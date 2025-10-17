/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 28, 2024.
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
#import "_WKUserContentExtensionStoreInternal.h"

#import "WKContentRuleListStoreInternal.h"
#import "WKContentRuleListStorePrivate.h"
#import "WKErrorInternal.h"
#import "_WKUserContentExtensionStorePrivate.h"
#import "_WKUserContentFilterInternal.h"
#import "_WKUserContentFilterPrivate.h"
#import <string>

NSString * const _WKUserContentExtensionsDomain = @"WKErrorDomain";

static NSError *toUserContentRuleListStoreError(const NSError *error)
{
    if (!error)
        return nil;

    ASSERT(error.domain == WKErrorDomain);
    switch (error.code) {
    case WKErrorContentRuleListStoreLookUpFailed:
        return [NSError errorWithDomain:_WKUserContentExtensionsDomain code:_WKUserContentExtensionStoreErrorLookupFailed userInfo:error.userInfo];
    case WKErrorContentRuleListStoreVersionMismatch:
        return [NSError errorWithDomain:_WKUserContentExtensionsDomain code:_WKUserContentExtensionStoreErrorVersionMismatch userInfo:error.userInfo];
    case WKErrorContentRuleListStoreCompileFailed:
        return [NSError errorWithDomain:_WKUserContentExtensionsDomain code:_WKUserContentExtensionStoreErrorCompileFailed userInfo:error.userInfo];
    case WKErrorContentRuleListStoreRemoveFailed:
        return [NSError errorWithDomain:_WKUserContentExtensionsDomain code:_WKUserContentExtensionStoreErrorRemoveFailed userInfo:error.userInfo];
    default:
        RELEASE_ASSERT_NOT_REACHED();
    }
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-implementations"
@implementation _WKUserContentExtensionStore
#pragma clang diagnostic pop

+ (instancetype)defaultStore
{
    return adoptNS([[_WKUserContentExtensionStore alloc] _initWithWKContentRuleListStore:[WKContentRuleListStore defaultStoreWithLegacyFilename]]).autorelease();
}

+ (instancetype)storeWithURL:(NSURL *)url
{
    return adoptNS([[_WKUserContentExtensionStore alloc] _initWithWKContentRuleListStore:[WKContentRuleListStore storeWithURLAndLegacyFilename:url]]).autorelease();
}

- (void)compileContentExtensionForIdentifier:(NSString *)identifier encodedContentExtension:(NSString *)encodedContentRuleList completionHandler:(void (^)(_WKUserContentFilter *, NSError *))completionHandler
{
    [_contentRuleListStore compileContentRuleListForIdentifier:identifier encodedContentRuleList:encodedContentRuleList completionHandler:^(WKContentRuleList *contentRuleList, NSError *error) {
        auto contentFilter = contentRuleList ? adoptNS([[_WKUserContentFilter alloc] _initWithWKContentRuleList:contentRuleList]) : nil;
        completionHandler(contentFilter.get(), toUserContentRuleListStoreError(error));
    }];
}

- (void)lookupContentExtensionForIdentifier:(NSString *)identifier completionHandler:(void (^)(_WKUserContentFilter *, NSError *))completionHandler
{
    [_contentRuleListStore lookUpContentRuleListForIdentifier:identifier completionHandler:^(WKContentRuleList *contentRuleList, NSError *error) {
        auto contentFilter = contentRuleList ? adoptNS([[_WKUserContentFilter alloc] _initWithWKContentRuleList:contentRuleList]) : nil;
        completionHandler(contentFilter.get(), toUserContentRuleListStoreError(error));
    }];
}

- (void)removeContentExtensionForIdentifier:(NSString *)identifier completionHandler:(void (^)(NSError *))completionHandler
{
    [_contentRuleListStore removeContentRuleListForIdentifier:identifier completionHandler:^(NSError *error) {
        completionHandler(toUserContentRuleListStoreError(error));
    }];
}

#pragma mark WKObject protocol implementation

- (API::Object&)_apiObject
{
    return [_contentRuleListStore _apiObject];
}

@end

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-implementations"
@implementation _WKUserContentExtensionStore (WKPrivate)
#pragma clang diagnostic pop

// For testing only.

- (void)_removeAllContentExtensions
{
    [_contentRuleListStore _removeAllContentRuleLists];
}

- (void)_invalidateContentExtensionVersionForIdentifier:(NSString *)identifier
{
    [_contentRuleListStore _invalidateContentRuleListVersionForIdentifier:identifier];
}

- (id)_initWithWKContentRuleListStore:(WKContentRuleListStore*)contentRuleListStore
{
    self = [super init];
    if (!self)
        return nil;
    
    _contentRuleListStore = contentRuleListStore;
    
    return self;
}

- (WKContentRuleListStore *)_contentRuleListStore
{
    return _contentRuleListStore.get();
}

@end
