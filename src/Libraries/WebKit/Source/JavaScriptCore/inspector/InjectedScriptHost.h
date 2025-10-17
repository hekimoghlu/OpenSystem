/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 11, 2024.
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

#include "JSCJSValueInlines.h"
#include "PerGlobalObjectWrapperWorld.h"
#include <wtf/RefCounted.h>

namespace Inspector {

class InjectedScriptHost : public RefCounted<InjectedScriptHost> {
public:
    static Ref<InjectedScriptHost> create() { return adoptRef(*new InjectedScriptHost); }
    JS_EXPORT_PRIVATE InjectedScriptHost();
    JS_EXPORT_PRIVATE virtual ~InjectedScriptHost();

    virtual JSC::JSValue subtype(JSC::JSGlobalObject*, JSC::JSValue) { return JSC::jsUndefined(); }
    virtual JSC::JSValue getInternalProperties(JSC::VM&, JSC::JSGlobalObject*, JSC::JSValue) { return { }; }
    virtual bool isHTMLAllCollection(JSC::VM&, JSC::JSValue) { return false; }

    JSC::JSValue wrapper(JSC::JSGlobalObject*);
    void clearAllWrappers();

    void setSavedResultAlias(const std::optional<String>& alias) { m_savedResultAlias = alias; }
    const std::optional<String>& savedResultAlias() const { return m_savedResultAlias; }

private:
    PerGlobalObjectWrapperWorld m_wrappers;
    std::optional<String> m_savedResultAlias;
};

} // namespace Inspector
