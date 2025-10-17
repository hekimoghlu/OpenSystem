/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 21, 2022.
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

#include "InjectedScriptBase.h"
#include <wtf/text/WTFString.h>

namespace JSC {
class JSValue;
}

namespace Inspector {

class InjectedScript;
class InjectedScriptManager;

class InjectedScriptModule : public InjectedScriptBase {
public:
    JS_EXPORT_PRIVATE ~InjectedScriptModule() override;
    virtual JSC::JSFunction* injectModuleFunction(JSC::JSGlobalObject*) const = 0;
    virtual JSC::JSValue host(InjectedScriptManager*, JSC::JSGlobalObject*) const = 0;

protected:
    // Do not expose constructor in the child classes as well. Instead provide
    // a static factory method that would create a new instance of the class
    // and call its ensureInjected() method immediately.
    JS_EXPORT_PRIVATE explicit InjectedScriptModule(const String& name);
    void ensureInjected(InjectedScriptManager*, JSC::JSGlobalObject*);
    JS_EXPORT_PRIVATE void ensureInjected(InjectedScriptManager*, const InjectedScript&);
};

} // namespace Inspector
