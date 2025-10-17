/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 28, 2025.
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

#include "BlobPropertyBag.h"
#include "BlobURL.h"
#include "FileReaderLoader.h"
#include "ScriptExecutionContext.h"
#include "ScriptWrappable.h"
#include "SecurityOriginData.h"
#include "URLKeepingBlobAlive.h"
#include "URLRegistry.h"
#include <variant>
#include <wtf/TZoneMalloc.h>
#include <wtf/URL.h>

namespace JSC {
class ArrayBufferView;
class ArrayBuffer;
}

namespace WebCore {

class Blob;
class BlobLoader;
class DeferredPromise;
class ReadableStream;
class ScriptExecutionContext;
class FragmentedSharedBuffer;
class WebCoreOpaqueRoot;

struct IDLArrayBuffer;

template<typename> class DOMPromiseDeferred;
template<typename> class ExceptionOr;

using BlobPartVariant = std::variant<RefPtr<JSC::ArrayBufferView>, RefPtr<JSC::ArrayBuffer>, RefPtr<Blob>, String>;

class Blob : public ScriptWrappable, public URLRegistrable, public RefCounted<Blob>, public ActiveDOMObject {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED_EXPORT(Blob, WEBCORE_EXPORT);
public:
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    static Ref<Blob> create(ScriptExecutionContext* context)
    {
        auto blob = adoptRef(*new Blob(context));
        blob->suspendIfNeeded();
        return blob;
    }

    static Ref<Blob> create(ScriptExecutionContext& context, Vector<BlobPartVariant>&& blobPartVariants, const BlobPropertyBag& propertyBag)
    {
        auto blob = adoptRef(*new Blob(context, WTFMove(blobPartVariants), propertyBag));
        blob->suspendIfNeeded();
        return blob;
    }

    static Ref<Blob> create(ScriptExecutionContext* context, Vector<uint8_t>&& data, const String& contentType)
    {
        auto blob = adoptRef(*new Blob(context, WTFMove(data), contentType));
        blob->suspendIfNeeded();
        return blob;
    }

    static Ref<Blob> deserialize(ScriptExecutionContext* context, const URL& srcURL, const String& type, long long size, long long memoryCost, const String& fileBackedPath)
    {
        ASSERT(Blob::isNormalizedContentType(type));
        auto blob = adoptRef(*new Blob(deserializationContructor, context, srcURL, type, size, memoryCost, fileBackedPath));
        blob->suspendIfNeeded();
        return blob;
    }

    virtual ~Blob();

    URL url() const { return m_internalURL; }
    const String& type() const { return m_type; }

    WEBCORE_EXPORT unsigned long long size() const;
    virtual bool isFile() const { return false; }

    // The checks described in the File API spec.
    static bool isValidContentType(const String&);
    // The normalization procedure described in the File API spec.
    static String normalizedContentType(const String&);
#if ASSERT_ENABLED
    static bool isNormalizedContentType(const String&);
    static bool isNormalizedContentType(const CString&);
#endif

    // URLRegistrable
    URLRegistry& registry() const final;
    RegistrableType registrableType() const final { return RegistrableType::Blob; }

    Ref<Blob> slice(long long start, long long end, const String& contentType) const;

    void text(Ref<DeferredPromise>&&);
    void arrayBuffer(DOMPromiseDeferred<IDLArrayBuffer>&&);
    void getArrayBuffer(CompletionHandler<void(ExceptionOr<Ref<JSC::ArrayBuffer>>)>&&);
    void bytes(Ref<DeferredPromise>&&);
    ExceptionOr<Ref<ReadableStream>> stream();

    size_t memoryCost() const { return m_memoryCost; }

    // Keeping the handle alive will keep the Blob data alive (but not the Blob object).
    URLKeepingBlobAlive handle() const;

protected:
    WEBCORE_EXPORT explicit Blob(ScriptExecutionContext*);
    Blob(ScriptExecutionContext&, Vector<BlobPartVariant>&&, const BlobPropertyBag&);
    Blob(ScriptExecutionContext*, Vector<uint8_t>&&, const String& contentType);

    enum ReferencingExistingBlobConstructor { referencingExistingBlobConstructor };
    Blob(ReferencingExistingBlobConstructor, ScriptExecutionContext*, const Blob&);

    enum UninitializedContructor { uninitializedContructor };
    Blob(UninitializedContructor, ScriptExecutionContext*, URL&&, String&& type);

    enum DeserializationContructor { deserializationContructor };
    Blob(DeserializationContructor, ScriptExecutionContext*, const URL& srcURL, const String& type, std::optional<unsigned long long> size, unsigned long long memoryCost, const String& fileBackedPath);

    // For slicing.
    Blob(ScriptExecutionContext*, const URL& srcURL, long long start, long long end, unsigned long long memoryCost, const String& contentType);

private:
    void loadBlob(FileReaderLoader::ReadType, Function<void(BlobLoader&)>&&);

    String m_type;
    mutable std::optional<unsigned long long> m_size;
    size_t m_memoryCost { 0 };

    // This is an internal URL referring to the blob data associated with this object. It serves
    // as an identifier for this blob. The internal URL is never used to source the blob's content
    // into an HTML or for FileRead'ing, public blob URLs must be used for those purposes.
    URL m_internalURL;

    UncheckedKeyHashSet<std::unique_ptr<BlobLoader>> m_blobLoaders;
};

WebCoreOpaqueRoot root(Blob*);

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::Blob)
    static bool isType(const WebCore::URLRegistrable& registrable) { return registrable.registrableType() == WebCore::URLRegistrable::RegistrableType::Blob; }
SPECIALIZE_TYPE_TRAITS_END()
