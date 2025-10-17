/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 18, 2025.
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

#include "GetPutInfo.h"
#include "JSObject.h"
#include "VariableEnvironment.h"

namespace JSC {

class ScopeChainIterator;
class SymbolTable;
class WatchpointSet;

using TDZEnvironment = UncheckedKeyHashSet<RefPtr<UniquedStringImpl>, IdentifierRepHash>;

class JSScope : public JSNonFinalObject {
public:
    using Base = JSNonFinalObject;
    static constexpr unsigned StructureFlags = Base::StructureFlags;

    template<typename, SubspaceAccess>
    static void subspaceFor(VM&)
    {
        RELEASE_ASSERT_NOT_REACHED();
    }

    DECLARE_EXPORT_INFO;

    friend class LLIntOffsetsExtractor;
    static size_t offsetOfNext();

    static JSObject* objectAtScope(JSScope*);

    static JSObject* resolve(JSGlobalObject*, JSScope*, const Identifier&);
    static JSValue resolveScopeForHoistingFuncDeclInEval(JSGlobalObject*, JSScope*, const Identifier&);
    static ResolveOp abstractResolve(JSGlobalObject*, size_t depthOffset, JSScope*, const Identifier&, GetOrPut, ResolveType, InitializationMode);

    static bool hasConstantScope(ResolveType);
    static JSScope* constantScopeForCodeBlock(ResolveType, CodeBlock*);

    static void collectClosureVariablesUnderTDZ(JSScope*, TDZEnvironment& result, PrivateNameEnvironment&);

    DECLARE_VISIT_CHILDREN;

    bool isVarScope();
    bool isLexicalScope();
    bool isModuleScope();
    bool isCatchScope();
    bool isCatchScopeWithSimpleParameter();
    bool isFunctionNameScopeObject();

    bool isNestedLexicalScope();

    ScopeChainIterator begin();
    ScopeChainIterator end();
    JSScope* next();

    JSObject* globalThis();

    SymbolTable* symbolTable();

protected:
    JSScope(VM&, Structure*, JSScope* next);

    template<typename ReturnPredicateFunctor, typename SkipPredicateFunctor>
    static JSObject* resolve(JSGlobalObject*, JSScope*, const Identifier&, ReturnPredicateFunctor, SkipPredicateFunctor);

private:
    WriteBarrier<JSScope> m_next;
};

inline JSScope::JSScope(VM& vm, Structure* structure, JSScope* next)
    : Base(vm, structure)
    , m_next(next, WriteBarrierEarlyInit)
{
}

class ScopeChainIterator {
public:
    ScopeChainIterator(JSScope* node)
        : m_node(node)
    {
    }

    JSObject* get() const { return JSScope::objectAtScope(m_node); }
    JSObject* operator->() const { return JSScope::objectAtScope(m_node); }
    JSScope* scope() const { return m_node; }

    ScopeChainIterator& operator++() { m_node = m_node->next(); return *this; }

    // postfix ++ intentionally omitted

    friend bool operator==(const ScopeChainIterator&, const ScopeChainIterator&) = default;

private:
    JSScope* m_node;
};

inline ScopeChainIterator JSScope::begin()
{
    return ScopeChainIterator(this); 
}

inline ScopeChainIterator JSScope::end()
{ 
    return ScopeChainIterator(nullptr); 
}

inline JSScope* JSScope::next()
{ 
    return m_next.get();
}

inline size_t JSScope::offsetOfNext()
{
    return OBJECT_OFFSETOF(JSScope, m_next);
}

} // namespace JSC
