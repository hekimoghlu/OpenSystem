/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 13, 2024.
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
#include "DFGLazyJSValue.h"

#if ENABLE(DFG_JIT)

#include "CCallHelpers.h"
#include "DFGGraph.h"
#include "JSCJSValueInlines.h"
#include "LinkBuffer.h"

namespace JSC { namespace DFG {

LazyJSValue LazyJSValue::newString(Graph& graph, const String& string)
{
    LazyJSValue result;
    result.m_kind = NewStringImpl;
    result.u.stringImpl = graph.m_localStrings.add(string).iterator->impl();
    return result;
}

JSValue LazyJSValue::getValue(VM& vm) const
{
    switch (m_kind) {
    case KnownValue:
        return value()->value();
    case SingleCharacterString:
        return jsSingleCharacterString(vm, u.character);
    case KnownStringImpl:
        return jsString(vm, String { u.stringImpl });
    case NewStringImpl:
        return jsString(vm, AtomStringImpl::add(u.stringImpl));
    }
    RELEASE_ASSERT_NOT_REACHED();
    return JSValue();
}

static TriState equalToSingleCharacter(JSValue value, UChar character)
{
    if (!value.isString())
        return TriState::False;
    
    JSString* jsString = asString(value);
    if (jsString->length() != 1)
        return TriState::False;
    
    const StringImpl* string = jsString->tryGetValueImpl();
    if (!string)
        return TriState::Indeterminate;
    
    return triState(string->at(0) == character);
}

static TriState equalToStringImpl(JSValue value, StringImpl* stringImpl)
{
    if (!value.isString())
        return TriState::False;
    
    JSString* jsString = asString(value);
    const StringImpl* string = jsString->tryGetValueImpl();
    if (!string)
        return TriState::Indeterminate;
    
    return triState(WTF::equal(stringImpl, string));
}

const StringImpl* LazyJSValue::tryGetStringImpl() const
{
    switch (m_kind) {
    case KnownStringImpl:
    case NewStringImpl:
        return u.stringImpl;

    case KnownValue:
        if (JSString* string = value()->dynamicCast<JSString*>())
            return string->tryGetValueImpl();
        return nullptr;

    case SingleCharacterString:
        return nullptr;
    }
    RELEASE_ASSERT_NOT_REACHED();
    return nullptr;
}

struct CrossThreadStringTranslator {
    static unsigned hash(const StringImpl* impl)
    {
        return impl->concurrentHash();
    }

    static bool equal(const String& string, const StringImpl* impl)
    {
        return WTF::equal(string.impl(), impl);
    }

    static void translate(String& location, const StringImpl* impl, unsigned)
    {
        location = impl->isolatedCopy();
    }
};

String LazyJSValue::tryGetString(Graph& graph) const
{
    switch (m_kind) {
    case NewStringImpl:
        return u.stringImpl;

    case SingleCharacterString:
        return span(u.character);

    case KnownValue:
    case KnownStringImpl:
        if (const StringImpl* string = tryGetStringImpl()) {
            unsigned ginormousStringLength = 10000;
            if (string->length() > ginormousStringLength)
                return String();
            
            return *graph.m_copiedStrings.add<CrossThreadStringTranslator>(string).iterator;
        }
        
        return String();
    }
    RELEASE_ASSERT_NOT_REACHED();
    return String();
}

TriState LazyJSValue::strictEqual(const LazyJSValue& other) const
{
    switch (m_kind) {
    case KnownValue:
        switch (other.m_kind) {
        case KnownValue: {
            if (!value()->value() || !other.value()->value())
                return value()->value() == other.value()->value() ? TriState::True : TriState::False;
            return JSValue::pureStrictEqual(value()->value(), other.value()->value());
        }
        case SingleCharacterString: {
            if (!value()->value())
                return TriState::False;
            return equalToSingleCharacter(value()->value(), other.character());
        }
        case KnownStringImpl:
        case NewStringImpl: {
            if (!value()->value())
                return TriState::False;
            return equalToStringImpl(value()->value(), other.stringImpl());
        }
        }
        break;
    case SingleCharacterString:
        switch (other.m_kind) {
        case SingleCharacterString:
            return triState(character() == other.character());
        case KnownStringImpl:
        case NewStringImpl:
            if (other.stringImpl()->length() != 1)
                return TriState::False;
            return triState(other.stringImpl()->at(0) == character());
        case KnownValue:
            return other.strictEqual(*this);
        }
        break;
    case KnownStringImpl:
    case NewStringImpl:
        switch (other.m_kind) {
        case KnownStringImpl:
        case NewStringImpl:
            return triState(WTF::equal(stringImpl(), other.stringImpl()));
        case SingleCharacterString:
        case KnownValue:
            return other.strictEqual(*this);
        }
        break;
    }
    RELEASE_ASSERT_NOT_REACHED();
    return TriState::False;
}

uintptr_t LazyJSValue::switchLookupValue(SwitchKind kind) const
{
    // NB. Not every kind of JSValue will be able to give you a switch lookup
    // value, and this method will assert, or do bad things, if you use it
    // for a kind of value that can't.
    switch (m_kind) {
    case KnownValue:
        switch (kind) {
        case SwitchImm:
            if (value()->value())
                return value()->value().asInt32();
            return 0;
        case SwitchCell:
            if (value()->value())
                return std::bit_cast<uintptr_t>(value()->value().asCell());
            return 0;
        default:
            RELEASE_ASSERT_NOT_REACHED();
            return 0;
        }
    case SingleCharacterString:
        switch (kind) {
        case SwitchChar:
            return character();
        default:
            RELEASE_ASSERT_NOT_REACHED();
            return 0;
        }
    case KnownStringImpl:
    case NewStringImpl:
        RELEASE_ASSERT_NOT_REACHED();
        return 0;
    }
    RELEASE_ASSERT_NOT_REACHED();
    return 0;
}

void LazyJSValue::emit(CCallHelpers& jit, JSValueRegs result, Plan& planRef) const
{
    if (m_kind == KnownValue) {
        jit.moveValue(value()->value(), result);
        return;
    }

    // It must be some kind of cell.
#if USE(JSVALUE32_64)
    jit.move(CCallHelpers::TrustedImm32(JSValue::CellTag), result.tagGPR());
#endif
    CCallHelpers::DataLabelPtr label = jit.moveWithPatch(
        CCallHelpers::TrustedImmPtr(static_cast<size_t>(0xd1e7beeflu)),
        result.payloadGPR());

    LazyJSValue thisValue = *this;

    // Once we do this, we're committed. Otherwise we leak memory. Note that we call ref/deref
    // manually to ensure that there is no concurrency shadiness. We are doing something here
    // that might be rather brutal: transfering ownership of this string.
    if (m_kind == NewStringImpl)
        thisValue.u.stringImpl->ref();

    CodeBlock* codeBlock = jit.codeBlock();
    
    auto* plan = &planRef;
    jit.addLinkTask([=] (LinkBuffer& linkBuffer) {
        auto patchLocation = linkBuffer.locationOf<JITCompilationPtrTag>(label);
        plan->addMainThreadFinalizationTask([=] {
            JSValue realValue = thisValue.getValue(codeBlock->vm());
            RELEASE_ASSERT(realValue.isCell());

            codeBlock->addConstant(ConcurrentJSLocker(codeBlock->m_lock), realValue);

            if (thisValue.m_kind == NewStringImpl)
                thisValue.u.stringImpl->deref();

            MacroAssembler::repatchPointer(patchLocation, realValue.asCell());
        });
    });
}

void LazyJSValue::dumpInContext(PrintStream& out, DumpContext* context) const
{
    switch (m_kind) {
    case KnownValue:
        value()->dumpInContext(out, context);
        return;
    case SingleCharacterString:
        out.print("Lazy:SingleCharacterString(");
        out.printf("%04X", static_cast<unsigned>(character()));
        out.print(" / ", StringImpl::utf8ForCharacters(span(u.character)).value(), ")");
        return;
    case KnownStringImpl:
        out.print("Lazy:KnownString(", stringImpl(), ")");
        return;
    case NewStringImpl:
        out.print("Lazy:NewString(", stringImpl(), ")");
        return;
    }
    RELEASE_ASSERT_NOT_REACHED();
}

void LazyJSValue::dump(PrintStream& out) const
{
    dumpInContext(out, nullptr);
}

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)

