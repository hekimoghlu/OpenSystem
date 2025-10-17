/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 8, 2023.
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
#include "ParserArena.h"

#include "CatchScope.h"
#include "JSBigInt.h"
#include "JSCInlines.h"
#include "Nodes.h"
#include "VMTrapsInlines.h"
#include <wtf/text/MakeString.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC {

DEFINE_ALLOCATOR_WITH_HEAP_IDENTIFIER(IdentifierArena);
DEFINE_ALLOCATOR_WITH_HEAP_IDENTIFIER(ParserArena);

ParserArena::ParserArena()
    : m_freeableMemory(nullptr)
    , m_freeablePoolEnd(nullptr)
{
}

inline void* ParserArena::freeablePool()
{
    ASSERT(m_freeablePoolEnd);
    return m_freeablePoolEnd - freeablePoolSize;
}

inline void ParserArena::deallocateObjects()
{
    size_t size = m_deletableObjects.size();
    for (size_t i = 0; i < size; ++i)
        m_deletableObjects[i]->~ParserArenaDeletable();

    if (m_freeablePoolEnd)
        ParserArenaMalloc::free(freeablePool());

    size = m_freeablePools.size();
    for (size_t i = 0; i < size; ++i)
        ParserArenaMalloc::free(m_freeablePools[i]);
}

ParserArena::~ParserArena()
{
    deallocateObjects();
}

void ParserArena::allocateFreeablePool()
{
    if (m_freeablePoolEnd)
        m_freeablePools.append(freeablePool());

    char* pool = static_cast<char*>(ParserArenaMalloc::malloc(freeablePoolSize));
    m_freeableMemory = pool;
    m_freeablePoolEnd = pool + freeablePoolSize;
    ASSERT(freeablePool() == pool);
}

const Identifier* IdentifierArena::makeBigIntDecimalIdentifier(VM& vm, const Identifier& identifier, uint8_t radix)
{
    if (radix == 10)
        return &identifier;

    DeferTermination deferScope(vm);
    auto scope = DECLARE_CATCH_SCOPE(vm);
    JSValue bigInt = JSBigInt::parseInt(nullptr, vm, identifier.string(), radix, JSBigInt::ErrorParseMode::ThrowExceptions, JSBigInt::ParseIntSign::Unsigned);
    scope.assertNoException();

    if (bigInt.isEmpty()) {
        // Handle out-of-memory or other failures by returning null, since
        // we don't have a global object to throw exceptions to in this scope.
        return nullptr;
    }

    // FIXME: We are allocating a JSBigInt just to be able to use
    // JSBigInt::tryGetString when radix is not 10.
    // This creates some GC pressure, but since these identifiers
    // will only be created when BigInt literal is used as a property name,
    // it wont be much problematic, given such cases are very rare.
    // There is a lot of optimizations we can apply here when necessary.
    // https://bugs.webkit.org/show_bug.cgi?id=207627
    JSBigInt* heapBigInt;
#if USE(BIGINT32)
    if (bigInt.isBigInt32()) {
        heapBigInt = JSBigInt::tryCreateFrom(vm, bigInt.bigInt32AsInt32());
        RELEASE_ASSERT(heapBigInt);
    } else
#endif
        heapBigInt = bigInt.asHeapBigInt();

    m_identifiers.append(Identifier::fromString(vm, JSBigInt::tryGetString(vm, heapBigInt, 10)));
    return &m_identifiers.last();
}

const Identifier& IdentifierArena::makePrivateIdentifier(VM& vm, ASCIILiteral prefix, unsigned identifier)
{
    auto symbolName = makeString(prefix, identifier);
    auto symbol = vm.privateSymbolRegistry().symbolForKey(symbolName);
    m_identifiers.append(Identifier::fromUid(symbol));
    return m_identifiers.last();
}

}

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
