/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 20, 2023.
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
#include "config.h"
#include "WKUserContentExtensionStoreRef.h"

#include "APIContentRuleList.h"
#include "APIContentRuleListStore.h"
#include "WKAPICast.h"
#include <wtf/CompletionHandler.h>

using namespace WebKit;

WKTypeID WKUserContentExtensionStoreGetTypeID()
{
#if ENABLE(CONTENT_EXTENSIONS)
    return toAPI(API::ContentRuleListStore::APIType);
#else
    return 0;
#endif
}

WKUserContentExtensionStoreRef WKUserContentExtensionStoreCreate(WKStringRef path)
{
#if ENABLE(CONTENT_EXTENSIONS)
    return toAPI(&API::ContentRuleListStore::storeWithPath(toWTFString(path)).leakRef());
#else
    UNUSED_PARAM(path);
    return nullptr;
#endif
}

#if ENABLE(CONTENT_EXTENSIONS)
static inline WKUserContentExtensionStoreResult toResult(const std::error_code& error)
{
    if (!error)
        return kWKUserContentExtensionStoreSuccess;

    switch (static_cast<API::ContentRuleListStore::Error>(error.value())) {
    case API::ContentRuleListStore::Error::LookupFailed:
        return kWKUserContentExtensionStoreLookupFailed;
    case API::ContentRuleListStore::Error::VersionMismatch:
        return kWKUserContentExtensionStoreVersionMismatch;
    case API::ContentRuleListStore::Error::CompileFailed:
        return kWKUserContentExtensionStoreCompileFailed;
    case API::ContentRuleListStore::Error::RemoveFailed:
        return kWKUserContentExtensionStoreRemoveFailed;
    }

    RELEASE_ASSERT_NOT_REACHED();
}
#endif

void WKUserContentExtensionStoreCompile(WKUserContentExtensionStoreRef store, WKStringRef identifier, WKStringRef jsonSource, void* context, WKUserContentExtensionStoreFunction callback)
{
#if ENABLE(CONTENT_EXTENSIONS)
    toImpl(store)->compileContentRuleList(toWTFString(identifier), toWTFString(jsonSource), [context, callback](RefPtr<API::ContentRuleList> contentRuleList, std::error_code error) {
        callback(error ? nullptr : toAPI(contentRuleList.leakRef()), toResult(error), context);
    });
#else
    UNUSED_PARAM(jsonSource);
    callback(nullptr, kWKUserContentExtensionStoreCompileFailed, context);
#endif
}

void WKUserContentExtensionStoreLookup(WKUserContentExtensionStoreRef store, WKStringRef identifier, void* context, WKUserContentExtensionStoreFunction callback)
{
#if ENABLE(CONTENT_EXTENSIONS)
    toImpl(store)->lookupContentRuleList(toWTFString(identifier), [context, callback](RefPtr<API::ContentRuleList> contentRuleList, std::error_code error) {
        callback(error ? nullptr : toAPI(contentRuleList.leakRef()), toResult(error), context);
    });
#else
    callback(nullptr, kWKUserContentExtensionStoreLookupFailed, context);
#endif
}

void WKUserContentExtensionStoreRemove(WKUserContentExtensionStoreRef store, WKStringRef identifier, void* context, WKUserContentExtensionStoreFunction callback)
{
#if ENABLE(CONTENT_EXTENSIONS)
    toImpl(store)->removeContentRuleList(toWTFString(identifier), [context, callback](std::error_code error) {
        callback(nullptr, toResult(error), context);
    });
#else
    callback(nullptr, kWKUserContentExtensionStoreRemoveFailed, context);
#endif
}
