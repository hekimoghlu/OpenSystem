/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 15, 2023.
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
#include "WKArray.h"

#include "APIArray.h"
#include "WKAPICast.h"
#include <wtf/StdLibExtras.h>

WKTypeID WKArrayGetTypeID()
{
    return WebKit::toAPI(API::Array::APIType);
}

WKArrayRef WKArrayCreate(WKTypeRef* rawValues, size_t numberOfValues)
{
    auto values = unsafeMakeSpan(rawValues, numberOfValues);
    Vector<RefPtr<API::Object>> elements(numberOfValues, [values](size_t i) -> RefPtr<API::Object> {
        return WebKit::toImpl(values[i]);
    });
    return WebKit::toAPI(&API::Array::create(WTFMove(elements)).leakRef());
}

WKArrayRef WKArrayCreateAdoptingValues(WKTypeRef* rawValues, size_t numberOfValues)
{
    auto values = unsafeMakeSpan(rawValues, numberOfValues);
    Vector<RefPtr<API::Object>> elements(numberOfValues, [values](size_t i) {
        return adoptRef(WebKit::toImpl(values[i]));
    });
    return WebKit::toAPI(&API::Array::create(WTFMove(elements)).leakRef());
}

WKTypeRef WKArrayGetItemAtIndex(WKArrayRef arrayRef, size_t index)
{
    return WebKit::toAPI(WebKit::toImpl(arrayRef)->at(index));
}

size_t WKArrayGetSize(WKArrayRef arrayRef)
{
    return WebKit::toImpl(arrayRef)->size();
}
