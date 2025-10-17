/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 21, 2022.
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
#include "WKData.h"

#include "APIData.h"
#include "WKAPICast.h"
#include <wtf/StdLibExtras.h>

WKTypeID WKDataGetTypeID()
{
    return WebKit::toAPI(API::Data::APIType);
}

WKDataRef WKDataCreate(const unsigned char* bytes, size_t size)
{
    return WebKit::toAPI(&API::Data::create(unsafeMakeSpan(bytes, size)).leakRef());
}

const unsigned char* WKDataGetBytes(WKDataRef dataRef)
{
    return WebKit::toImpl(dataRef)->span().data();
}

size_t WKDataGetSize(WKDataRef dataRef)
{
    return WebKit::toImpl(dataRef)->size();
}

std::span<const uint8_t> WKDataGetSpan(WKDataRef dataRef)
{
    return unsafeMakeSpan(byteCast<uint8_t>(WKDataGetBytes(dataRef)), WKDataGetSize(dataRef));
}
