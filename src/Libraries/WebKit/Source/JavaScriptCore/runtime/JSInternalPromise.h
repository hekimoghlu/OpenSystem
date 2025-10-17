/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 2, 2024.
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
#pragma once

#include "JSPromise.h"

namespace JSC {

class JSFunction;

// JSInternalPromise is completely separated instance from the JSPromise.
// Since its prototype and constructor are different from the exposed Promises' ones,
// all the user modification onto the exposed Promise does not have effect on JSInternalPromise.
//
// e.g.
//     Replacing Promise.prototype.then with the user-customized one does not effect on JSInternalPromise.
//
// CAUTION: Must not leak the JSInternalPromise to the user space to keep its integrity.
class JSInternalPromise final : public JSPromise {
public:
    typedef JSPromise Base;

    JS_EXPORT_PRIVATE static JSInternalPromise* create(VM&, Structure*);
    static JSInternalPromise* createWithInitialValues(VM&, Structure*);
    static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    DECLARE_EXPORT_INFO;

    JS_EXPORT_PRIVATE JSInternalPromise* then(JSGlobalObject*, JSFunction* = nullptr, JSFunction* = nullptr);

    JS_EXPORT_PRIVATE JSInternalPromise* rejectWithCaughtException(JSGlobalObject*, ThrowScope&);

private:
    JSInternalPromise(VM&, Structure*);
};

} // namespace JSC
