/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 15, 2023.
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
#include "JSONParseTest.h"

#include "JSCInlines.h"
#include "JSGlobalObject.h"
#include "JSONObject.h"
#include "VM.h"
#include <wtf/RefPtr.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

using namespace JSC;

int testJSONParse()
{
    bool failed = false;

    RefPtr<VM> vm = VM::create();
    
    JSLockHolder locker(vm.get());
    JSGlobalObject* globalObject = JSGlobalObject::create(*vm, JSGlobalObject::createStructure(*vm, jsNull()));
    
    JSValue v0 = JSONParse(globalObject, ""_s);
    JSValue v1 = JSONParse(globalObject, "#$%^"_s);
    JSValue v2 = JSONParse(globalObject, String());
    UChar emptyUCharArray[1] = { '\0' };
    unsigned zeroLength = 0;
    JSValue v3 = JSONParse(globalObject, String({ emptyUCharArray, zeroLength }));
    JSValue v4;
    JSValue v5 = JSONParse(globalObject, "123"_s);
    
    failed = failed || (v0 != v1);
    failed = failed || (v1 != v2);
    failed = failed || (v2 != v3);
    failed = failed || (v3 != v4);
    failed = failed || (v4 == v5);

    vm = nullptr;

    if (failed)
        printf("FAIL: JSONParse String test.\n");
    else
        printf("PASS: JSONParse String test.\n");

    return failed;
}

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
