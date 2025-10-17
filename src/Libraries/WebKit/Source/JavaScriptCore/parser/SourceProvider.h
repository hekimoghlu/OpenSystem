/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 1, 2022.
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

#include <wtf/Compiler.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

#include "CachedBytecode.h"
#include "CodeSpecializationKind.h"
#include "SourceOrigin.h"
#include "SourceTaintedOrigin.h"
#include <wtf/RefCounted.h>
#include <wtf/text/TextPosition.h>
#include <wtf/text/WTFString.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

namespace JSC {

class SourceCode;
class UnlinkedFunctionExecutable;
class UnlinkedFunctionCodeBlock;

    enum class SourceProviderSourceType : uint8_t {
        Program,
        Module,
        WebAssembly,
        JSON,
        ImportMap,
    };

    using BytecodeCacheGenerator = Function<RefPtr<CachedBytecode>()>;

    class SourceProvider : public RefCounted<SourceProvider> {
    public:
        static const intptr_t nullID = 1;
        
        JS_EXPORT_PRIVATE SourceProvider(const SourceOrigin&, String&& sourceURL, String&& preRedirectURL, SourceTaintedOrigin, const TextPosition& startPosition, SourceProviderSourceType);

        JS_EXPORT_PRIVATE virtual ~SourceProvider();

        virtual unsigned hash() const = 0;
        virtual StringView source() const = 0;
        virtual RefPtr<CachedBytecode> cachedBytecode() const { return nullptr; }
        virtual void cacheBytecode(const BytecodeCacheGenerator&) const { }
        virtual void updateCache(const UnlinkedFunctionExecutable*, const SourceCode&, CodeSpecializationKind, const UnlinkedFunctionCodeBlock*) const { }
        virtual void commitCachedBytecode() const { }

        StringView getRange(int start, int end) const
        {
            return source().substring(start, end - start);
        }

        const SourceOrigin& sourceOrigin() const { return m_sourceOrigin; }

        // This is NOT the path that should be used for computing relative paths from a script. Use SourceOrigin's URL for that, the values may or may not be the same...
        const String& sourceURL() const { return m_sourceURL; }
        const String& sourceURLStripped();
        const String& preRedirectURL() const { return m_preRedirectURL; }
        const String& sourceURLDirective() const { return m_sourceURLDirective; }
        const String& sourceMappingURLDirective() const { return m_sourceMappingURLDirective; }

        TextPosition startPosition() const { return m_startPosition; }
        SourceProviderSourceType sourceType() const { return m_sourceType; }

        SourceID asID()
        {
            if (!m_id)
                getID();
            return m_id;
        }

        void setSourceURLDirective(const String& sourceURLDirective) { m_sourceURLDirective = sourceURLDirective; }
        void setSourceMappingURLDirective(const String& sourceMappingURLDirective) { m_sourceMappingURLDirective = sourceMappingURLDirective; }
        void setSourceTaintedOrigin(SourceTaintedOrigin taintedness) { m_taintedness = taintedness; }

        SourceTaintedOrigin sourceTaintedOrigin() const { return m_taintedness; }
        bool couldBeTainted() const { return m_taintedness != SourceTaintedOrigin::Untainted; }

    private:
        JS_EXPORT_PRIVATE void getID();

        SourceProviderSourceType m_sourceType;
        SourceOrigin m_sourceOrigin;
        String m_sourceURL;
        String m_sourceURLStripped;
        String m_preRedirectURL;
        String m_sourceURLDirective;
        String m_sourceMappingURLDirective;
        TextPosition m_startPosition;
        SourceID m_id { 0 };
        SourceTaintedOrigin m_taintedness;
    };

    DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER(StringSourceProvider);
    class StringSourceProvider : public SourceProvider {
        WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(StringSourceProvider);
    public:
        static Ref<StringSourceProvider> create(const String& source, const SourceOrigin& sourceOrigin, String sourceURL, SourceTaintedOrigin taintedness, const TextPosition& startPosition = TextPosition(), SourceProviderSourceType sourceType = SourceProviderSourceType::Program)
        {
            return adoptRef(*new StringSourceProvider(source, sourceOrigin, taintedness, WTFMove(sourceURL), startPosition, sourceType));
        }
        
        unsigned hash() const override
        {
            return m_source.get().hash();
        }

        StringView source() const override
        {
            return m_source.get();
        }

    protected:
        StringSourceProvider(const String& source, const SourceOrigin& sourceOrigin, SourceTaintedOrigin taintedness, String&& sourceURL, const TextPosition& startPosition, SourceProviderSourceType sourceType)
            : SourceProvider(sourceOrigin, WTFMove(sourceURL), String(), taintedness, startPosition, sourceType)
            , m_source(source.isNull() ? *StringImpl::empty() : *source.impl())
        {
        }

    private:
        Ref<StringImpl> m_source;
    };

#if ENABLE(WEBASSEMBLY)
    class BaseWebAssemblySourceProvider : public SourceProvider {
    public:
        virtual const uint8_t* data() = 0;
        virtual size_t size() const = 0;
        virtual void lockUnderlyingBuffer() { }
        virtual void unlockUnderlyingBuffer() { }

    protected:
        JS_EXPORT_PRIVATE BaseWebAssemblySourceProvider(const SourceOrigin&, String&& sourceURL);
    };

    class WebAssemblySourceProvider final : public BaseWebAssemblySourceProvider {
    public:
        static Ref<WebAssemblySourceProvider> create(Vector<uint8_t>&& data, const SourceOrigin& sourceOrigin, String sourceURL)
        {
            return adoptRef(*new WebAssemblySourceProvider(WTFMove(data), sourceOrigin, WTFMove(sourceURL)));
        }

        unsigned hash() const final
        {
            return m_source.impl()->hash();
        }

        StringView source() const final
        {
            return m_source;
        }

        const uint8_t* data() final
        {
            return m_data.data();
        }

        size_t size() const final
        {
            return m_data.size();
        }

        const Vector<uint8_t>& dataVector() const
        {
            return m_data;
        }

    private:
        WebAssemblySourceProvider(Vector<uint8_t>&& data, const SourceOrigin& sourceOrigin, String&& sourceURL)
            : BaseWebAssemblySourceProvider(sourceOrigin, WTFMove(sourceURL))
            , m_source("[WebAssembly source]"_s)
            , m_data(WTFMove(data))
        {
        }

        String m_source;
        Vector<uint8_t> m_data;
    };

    // RAII class for managing a Wasm source provider's underlying buffer.
    class WebAssemblySourceProviderBufferGuard {
    public:
        explicit WebAssemblySourceProviderBufferGuard(BaseWebAssemblySourceProvider* sourceProvider)
            : m_sourceProvider(sourceProvider)
        {
            if (m_sourceProvider)
                m_sourceProvider->lockUnderlyingBuffer();
        }

        ~WebAssemblySourceProviderBufferGuard()
        {
            if (m_sourceProvider)
                m_sourceProvider->unlockUnderlyingBuffer();
        }

    private:
        RefPtr<BaseWebAssemblySourceProvider> m_sourceProvider;
    };
#endif

} // namespace JSC
