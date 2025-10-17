/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 8, 2022.
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

#include "APICast.h"
#include "Completion.h"
#include "Exception.h"
#include "JSGlobalObjectInlines.h"
#include "JSScriptRefPrivate.h"
#include "OpaqueJSString.h"
#include "Parser.h"
#include "SourceCode.h"
#include "SourceProvider.h"

using namespace JSC;

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

struct OpaqueJSScript final : public SourceProvider {
public:
    static WTF::Ref<OpaqueJSScript> create(VM& vm, const SourceOrigin& sourceOrigin, String filename, int startingLineNumber, const String& source)
    {
        return WTF::adoptRef(*new OpaqueJSScript(vm, sourceOrigin, WTFMove(filename), startingLineNumber, source));
    }

    unsigned hash() const final
    {
        return m_source.get().hash();
    }

    StringView source() const final
    {
        return m_source.get();
    }

    VM& vm() const { return m_vm; }

private:
    OpaqueJSScript(VM& vm, const SourceOrigin& sourceOrigin, String&& filename, int startingLineNumber, const String& source)
        : SourceProvider(sourceOrigin, WTFMove(filename), String(), SourceTaintedOrigin::Untainted, TextPosition(OrdinalNumber::fromOneBasedInt(startingLineNumber), OrdinalNumber()), SourceProviderSourceType::Program)
        , m_vm(vm)
        , m_source(source.isNull() ? *StringImpl::empty() : *source.impl())
    {
    }

    ~OpaqueJSScript() final { }

    VM& m_vm;
    Ref<StringImpl> m_source;
};

static bool parseScript(VM& vm, const SourceCode& source, ParserError& error)
{
    return !!parseRootNode<ProgramNode>(
        vm, source, ImplementationVisibility::Public, JSParserBuiltinMode::NotBuiltin,
        NoLexicallyScopedFeatures, JSParserScriptMode::Classic, SourceParseMode::ProgramMode, error);
}

extern "C" {

JSScriptRef JSScriptCreateReferencingImmortalASCIIText(JSContextGroupRef contextGroup, JSStringRef urlString, int startingLineNumber, const char* source, size_t length, JSStringRef* errorMessage, int* errorLine)
{
    auto& vm = *toJS(contextGroup);
    JSLockHolder locker(&vm);
    for (size_t i = 0; i < length; i++) {
        if (!isASCII(source[i]))
            return nullptr;
    }

    startingLineNumber = std::max(1, startingLineNumber);

    auto sourceURL = urlString ? URL({ }, urlString->string()) : URL();
    auto result = OpaqueJSScript::create(vm, SourceOrigin { sourceURL }, sourceURL.string(), startingLineNumber, String(StringImpl::createWithoutCopying({ source, length })));

    ParserError error;
    if (!parseScript(vm, SourceCode(result.copyRef()), error)) {
        if (errorMessage)
            *errorMessage = OpaqueJSString::tryCreate(error.message()).leakRef();
        if (errorLine)
            *errorLine = error.line();
        return nullptr;
    }

    return &result.leakRef();
}

JSScriptRef JSScriptCreateFromString(JSContextGroupRef contextGroup, JSStringRef urlString, int startingLineNumber, JSStringRef source, JSStringRef* errorMessage, int* errorLine)
{
    auto& vm = *toJS(contextGroup);
    JSLockHolder locker(&vm);

    startingLineNumber = std::max(1, startingLineNumber);

    auto sourceURL = urlString ? URL({ }, urlString->string()) : URL();
    auto result = OpaqueJSScript::create(vm, SourceOrigin { sourceURL }, sourceURL.string(), startingLineNumber, source->string());

    ParserError error;
    if (!parseScript(vm, SourceCode(result.copyRef()), error)) {
        if (errorMessage)
            *errorMessage = OpaqueJSString::tryCreate(error.message()).leakRef();
        if (errorLine)
            *errorLine = error.line();
        return nullptr;
    }

    return &result.leakRef();
}

void JSScriptRetain(JSScriptRef script)
{
    JSLockHolder locker(&script->vm());
    script->ref();
}

void JSScriptRelease(JSScriptRef script)
{
    JSLockHolder locker(&script->vm());
    script->deref();
}

JSValueRef JSScriptEvaluate(JSContextRef context, JSScriptRef script, JSValueRef thisValueRef, JSValueRef* exception)
{
    JSGlobalObject* globalObject = toJS(context);
    VM& vm = globalObject->vm();
    JSLockHolder locker(vm);
    if (&script->vm() != &vm) {
        RELEASE_ASSERT_NOT_REACHED();
        return nullptr;
    }
    NakedPtr<Exception> internalException;
    JSValue thisValue = thisValueRef ? toJS(globalObject, thisValueRef) : jsUndefined();
    JSValue result = evaluate(globalObject, SourceCode(*script), thisValue, internalException);
    if (internalException) {
        if (exception)
            *exception = toRef(globalObject, internalException->value());
        return nullptr;
    }
    ASSERT(result);
    return toRef(globalObject, result);
}

}

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
