/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 20, 2024.
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
#include "WKNumber.h"

#include "APINumber.h"
#include "WKAPICast.h"

WKTypeID WKBooleanGetTypeID()
{
    return WebKit::toAPI(API::Boolean::APIType);
}

WKBooleanRef WKBooleanCreate(bool value)
{
    auto booleanObject = API::Boolean::create(value);
    return WebKit::toAPI(&booleanObject.leakRef());
}

bool WKBooleanGetValue(WKBooleanRef booleanRef)
{
    return WebKit::toImpl(booleanRef)->value();
}

WKTypeID WKDoubleGetTypeID()
{
    return WebKit::toAPI(API::Double::APIType);
}

WKDoubleRef WKDoubleCreate(double value)
{
    auto doubleObject = API::Double::create(value);
    return WebKit::toAPI(&doubleObject.leakRef());
}

double WKDoubleGetValue(WKDoubleRef doubleRef)
{
    return WebKit::toImpl(doubleRef)->value();
}

WKTypeID WKUInt64GetTypeID()
{
    return WebKit::toAPI(API::UInt64::APIType);
}

WKUInt64Ref WKUInt64Create(uint64_t value)
{
    auto uint64Object = API::UInt64::create(value);
    return WebKit::toAPI(&uint64Object.leakRef());
}

uint64_t WKUInt64GetValue(WKUInt64Ref uint64Ref)
{
    return WebKit::toImpl(uint64Ref)->value();
}
