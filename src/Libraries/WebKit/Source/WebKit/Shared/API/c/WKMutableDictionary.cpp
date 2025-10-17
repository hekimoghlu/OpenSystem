/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 13, 2025.
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
#include "WKMutableDictionary.h"

#include "APIDictionary.h"
#include "WKAPICast.h"

WKMutableDictionaryRef WKMutableDictionaryCreate()
{
    return const_cast<WKMutableDictionaryRef>(WebKit::toAPI(&API::Dictionary::create().leakRef()));
}

WKMutableDictionaryRef WKMutableDictionaryCreateWithCapacity(size_t capacity)
{
    return const_cast<WKMutableDictionaryRef>(WebKit::toAPI(&API::Dictionary::createWithCapacity(capacity).leakRef()));
}

bool WKDictionarySetItem(WKMutableDictionaryRef dictionaryRef, WKStringRef keyRef, WKTypeRef itemRef)
{
    return WebKit::toImpl(dictionaryRef)->set(WebKit::toImpl(keyRef)->string(), WebKit::toImpl(itemRef));
}

