/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 8, 2024.
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
#include "WKDictionary.h"

#include "APIArray.h"
#include "APIDictionary.h"
#include "WKAPICast.h"

WKTypeID WKDictionaryGetTypeID()
{
    return WebKit::toAPI(API::Dictionary::APIType);
}

WK_EXPORT WKDictionaryRef WKDictionaryCreate(const WKStringRef* rawKeys, const WKTypeRef* rawValues, size_t numberOfValues)
{
    auto keys = unsafeMakeSpan(rawKeys, numberOfValues);
    auto values = unsafeMakeSpan(rawValues, numberOfValues);

    API::Dictionary::MapType map;
    map.reserveInitialCapacity(numberOfValues);
    for (size_t i = 0; i < numberOfValues; ++i)
        map.add(WebKit::toImpl(keys[i])->string(), WebKit::toImpl(values[i]));

    return WebKit::toAPI(&API::Dictionary::create(WTFMove(map)).leakRef());
}

WKTypeRef WKDictionaryGetItemForKey(WKDictionaryRef dictionaryRef, WKStringRef key)
{
    return WebKit::toAPI(WebKit::toImpl(dictionaryRef)->get(WebKit::toImpl(key)->string()));
}

size_t WKDictionaryGetSize(WKDictionaryRef dictionaryRef)
{
    return WebKit::toImpl(dictionaryRef)->size();
}

WKArrayRef WKDictionaryCopyKeys(WKDictionaryRef dictionaryRef)
{
    return WebKit::toAPI(&WebKit::toImpl(dictionaryRef)->keys().leakRef());
}
