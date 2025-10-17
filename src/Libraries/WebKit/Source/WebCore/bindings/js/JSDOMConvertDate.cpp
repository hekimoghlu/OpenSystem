/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 20, 2023.
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
#include "JSDOMConvertDate.h"

#include <JavaScriptCore/DateInstance.h>
#include <JavaScriptCore/HeapInlines.h>
#include <JavaScriptCore/JSCJSValueInlines.h>

namespace WebCore {
using namespace JSC;

JSValue jsDate(JSGlobalObject& lexicalGlobalObject, WallTime value)
{
    return DateInstance::create(lexicalGlobalObject.vm(), lexicalGlobalObject.dateStructure(), value.secondsSinceEpoch().milliseconds());
}

WallTime valueToDate(JSC::JSGlobalObject& lexicalGlobalObject, JSValue value)
{
    double milliseconds = std::numeric_limits<double>::quiet_NaN();

    auto& vm = lexicalGlobalObject.vm();
    if (value.inherits<DateInstance>())
        milliseconds = jsCast<DateInstance*>(value)->internalNumber();
    else if (value.isNumber())
        milliseconds = value.asNumber();
    else if (value.isString())
        milliseconds = vm.dateCache.parseDate(&lexicalGlobalObject, vm, value.getString(&lexicalGlobalObject));

    return WallTime::fromRawSeconds(Seconds::fromMilliseconds(milliseconds).value());
}

} // namespace WebCore
