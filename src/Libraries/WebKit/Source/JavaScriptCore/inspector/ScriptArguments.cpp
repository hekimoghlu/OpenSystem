/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 19, 2025.
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
#include "ScriptArguments.h"

#include "CatchScope.h"
#include "ProxyObject.h"
#include "StrongInlines.h"

namespace Inspector {

static inline String argumentAsString(JSC::JSGlobalObject* globalObject, JSC::JSValue argument)
{
    if (JSC::jsDynamicCast<JSC::ProxyObject*>(argument))
        return "[object Proxy]"_s;

    auto scope = DECLARE_CATCH_SCOPE(globalObject->vm());
    auto result = argument.toWTFString(globalObject);
    scope.clearException();
    return result;
}

Ref<ScriptArguments> ScriptArguments::create(JSC::JSGlobalObject* globalObject, Vector<JSC::Strong<JSC::Unknown>>&& arguments)
{
    return adoptRef(*new ScriptArguments(globalObject, WTFMove(arguments)));
}

ScriptArguments::ScriptArguments(JSC::JSGlobalObject* globalObject, Vector<JSC::Strong<JSC::Unknown>>&& arguments)
    : m_globalObject(globalObject->vm(), globalObject)
    , m_arguments(WTFMove(arguments))
{
}

ScriptArguments::~ScriptArguments() = default;

JSC::JSValue ScriptArguments::argumentAt(size_t index) const
{
    ASSERT(m_arguments.size() > index);
    return m_arguments[index].get();
}

JSC::JSGlobalObject* ScriptArguments::globalObject() const
{
    return m_globalObject.get();
}

std::optional<String> ScriptArguments::getArgumentAtIndexAsString(size_t argumentIndex) const
{
    if (argumentIndex >= argumentCount())
        return std::nullopt;

    auto* globalObject = this->globalObject();
    if (!globalObject) {
        ASSERT_NOT_REACHED();
        return std::nullopt;
    }

    return argumentAsString(globalObject, argumentAt(argumentIndex));
}

bool ScriptArguments::getFirstArgumentAsString(String& result) const
{
    auto argument = getArgumentAtIndexAsString(0);
    if (!argument)
        return false;

    result = *argument;
    return true;
}

Vector<String> ScriptArguments::getArgumentsAsStrings() const
{
    auto* globalObject = this->globalObject();
    ASSERT(globalObject);
    if (!globalObject)
        return { };

    return WTF::map(m_arguments, [globalObject](auto& argument) {
        return argumentAsString(globalObject, argument.get());
    });
}

bool ScriptArguments::isEqual(const ScriptArguments& other) const
{
    auto size = m_arguments.size();

    if (size != other.m_arguments.size())
        return false;

    if (!size)
        return true;

    auto* globalObject = this->globalObject();
    if (!globalObject)
        return false;

    for (size_t i = 0; i < size; ++i) {
        auto a = m_arguments[i].get();
        auto b = other.m_arguments[i].get();
        if (!a || !b) {
            if (a != b)
                return false;
        } else {
            auto scope = DECLARE_CATCH_SCOPE(globalObject->vm());
            bool result = JSC::JSValue::strictEqual(globalObject, a, b);
            scope.clearException();
            if (!result)
                return false;
        }
    }

    return true;
}

} // namespace Inspector
