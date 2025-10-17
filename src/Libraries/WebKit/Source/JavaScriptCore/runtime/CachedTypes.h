/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 25, 2024.
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

#include "JSCast.h"
#include "ParserModes.h"
#include "VariableEnvironment.h"
#include <wtf/FileSystem.h>
#include <wtf/HashMap.h>

namespace JSC {

class BytecodeCacheError;
class CachedBytecode;
class SourceCodeKey;
class UnlinkedCodeBlock;
class UnlinkedFunctionCodeBlock;
class UnlinkedFunctionExecutable;

enum class SourceCodeType;

// This struct has to be updated when incrementally writing to the bytecode
// cache, since this will only be filled in when we parse the function
struct CachedFunctionExecutableMetadata {
    CodeFeatures m_features;
    LexicallyScopedFeatures m_lexicallyScopedFeatures;
    bool m_hasCapturedVariables;
};

struct CachedFunctionExecutableOffsets {
    static ptrdiff_t codeBlockForCallOffset();
    static ptrdiff_t codeBlockForConstructOffset();
    static ptrdiff_t metadataOffset();
};

struct CachedWriteBarrierOffsets {
    static ptrdiff_t ptrOffset();
};

struct CachedPtrOffsets {
    static ptrdiff_t offsetOffset();
};

class VariableLengthObjectBase {
    friend class CachedBytecode;

protected:
    VariableLengthObjectBase(ptrdiff_t offset)
        : m_offset(offset)
    {
    }

    ptrdiff_t m_offset;
};

class Decoder : public RefCounted<Decoder> {
    WTF_MAKE_NONCOPYABLE(Decoder);

public:
    static Ref<Decoder> create(VM&, Ref<CachedBytecode>, RefPtr<SourceProvider> = nullptr);

    ~Decoder();

    VM& vm() { return m_vm; }
    size_t size() const;

    ptrdiff_t offsetOf(const void*);
    void cacheOffset(ptrdiff_t, void*);
    std::optional<void*> cachedPtrForOffset(ptrdiff_t);
    const void* ptrForOffsetFromBase(ptrdiff_t);
    CompactTDZEnvironmentMap::Handle handleForTDZEnvironment(CompactTDZEnvironment*) const;
    void setHandleForTDZEnvironment(CompactTDZEnvironment*, const CompactTDZEnvironmentMap::Handle&);
    void addLeafExecutable(const UnlinkedFunctionExecutable*, ptrdiff_t);
    RefPtr<SourceProvider> provider() const;

    template<typename Functor>
    void addFinalizer(const Functor&);

private:
    Decoder(VM&, Ref<CachedBytecode>, RefPtr<SourceProvider>);

    VM& m_vm;
    Ref<CachedBytecode> m_cachedBytecode;
    UncheckedKeyHashMap<ptrdiff_t, void*> m_offsetToPtrMap;
    Vector<std::function<void()>> m_finalizers;
    UncheckedKeyHashMap<CompactTDZEnvironment*, CompactTDZEnvironmentMap::Handle> m_environmentToHandleMap;
    RefPtr<SourceProvider> m_provider;
};

JS_EXPORT_PRIVATE RefPtr<CachedBytecode> encodeCodeBlock(VM&, const SourceCodeKey&, const UnlinkedCodeBlock*);
JS_EXPORT_PRIVATE RefPtr<CachedBytecode> encodeCodeBlock(VM&, const SourceCodeKey&, const UnlinkedCodeBlock*, FileSystem::PlatformFileHandle fd, BytecodeCacheError&);

UnlinkedCodeBlock* decodeCodeBlockImpl(VM&, const SourceCodeKey&, Ref<CachedBytecode>);

template<typename UnlinkedCodeBlockType>
UnlinkedCodeBlockType* decodeCodeBlock(VM& vm, const SourceCodeKey& key, Ref<CachedBytecode> cachedBytecode)
{
    return jsCast<UnlinkedCodeBlockType*>(decodeCodeBlockImpl(vm, key, WTFMove(cachedBytecode)));
}

JS_EXPORT_PRIVATE RefPtr<CachedBytecode> encodeFunctionCodeBlock(VM&, const UnlinkedFunctionCodeBlock*, BytecodeCacheError&);

JS_EXPORT_PRIVATE void decodeFunctionCodeBlock(Decoder&, int32_t cachedFunctionCodeBlockOffset, WriteBarrier<UnlinkedFunctionCodeBlock>&, const JSCell*);

bool isCachedBytecodeStillValid(VM&, Ref<CachedBytecode>, const SourceCodeKey&, SourceCodeType);

} // namespace JSC
