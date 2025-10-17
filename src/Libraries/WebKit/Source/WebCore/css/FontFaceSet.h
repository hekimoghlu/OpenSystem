/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 27, 2025.
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

#include "ActiveDOMObject.h"
#include "CSSFontFaceSet.h"
#include "EventTarget.h"
#include "IDLTypes.h"
#include <wtf/UniqueRef.h>

namespace WebCore {

template<typename IDLType> class DOMPromiseDeferred;
template<typename IDLType> class DOMPromiseProxyWithResolveCallback;

class DOMException;

class FontFaceSet final : public RefCounted<FontFaceSet>, private FontEventClient, public EventTarget, public ActiveDOMObject {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(FontFaceSet);
public:
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    static Ref<FontFaceSet> create(ScriptExecutionContext&, const Vector<Ref<FontFace>>& initialFaces);
    static Ref<FontFaceSet> create(ScriptExecutionContext&, CSSFontFaceSet& backing);
    virtual ~FontFaceSet();

    bool has(FontFace&) const;
    size_t size();
    ExceptionOr<FontFaceSet&> add(FontFace&);
    bool remove(FontFace&);
    void clear();

    using LoadPromise = DOMPromiseDeferred<IDLSequence<IDLInterface<FontFace>>>;
    void load(const String& font, const String& text, LoadPromise&&);
    ExceptionOr<bool> check(const String& font, const String& text);

    enum class LoadStatus { Loading, Loaded };
    LoadStatus status() const;

    using ReadyPromise = DOMPromiseProxyWithResolveCallback<IDLInterface<FontFaceSet>>;
    ReadyPromise& ready() { return m_readyPromise.get(); }
    void documentDidFinishLoading();

    CSSFontFaceSet& backing() { return m_backing; }

    class Iterator {
    public:
        explicit Iterator(FontFaceSet&);
        RefPtr<FontFace> next();

    private:
        Ref<FontFaceSet> m_target;
        size_t m_index { 0 }; // FIXME: There needs to be a mechanism to handle when fonts are added or removed from the middle of the FontFaceSet.
    };
    Iterator createIterator(ScriptExecutionContext*) { return Iterator(*this); }

private:
    struct PendingPromise : RefCounted<PendingPromise> {
        static Ref<PendingPromise> create(LoadPromise&& promise)
        {
            return adoptRef(*new PendingPromise(WTFMove(promise)));
        }
        ~PendingPromise();

    private:
        PendingPromise(LoadPromise&&);

    public:
        Vector<Ref<FontFace>> faces;
        UniqueRef<LoadPromise> promise;
        bool hasReachedTerminalState { false };
    };

    FontFaceSet(ScriptExecutionContext&, const Vector<Ref<FontFace>>&);
    FontFaceSet(ScriptExecutionContext&, CSSFontFaceSet&);

    // FontEventClient
    void faceFinished(CSSFontFace&, CSSFontFace::Status) final;
    void startedLoading() final;
    void completedLoading() final;

    // EventTarget
    enum EventTargetInterfaceType eventTargetInterface() const final { return EventTargetInterfaceType::FontFaceSet; }
    ScriptExecutionContext* scriptExecutionContext() const final { return ActiveDOMObject::scriptExecutionContext(); }
    void refEventTarget() final { ref(); }
    void derefEventTarget() final { deref(); }

    // Callback for ReadyPromise.
    FontFaceSet& readyPromiseResolve();

    Ref<CSSFontFaceSet> m_backing;
    UncheckedKeyHashMap<RefPtr<FontFace>, Vector<Ref<PendingPromise>>> m_pendingPromises;
    UniqueRef<ReadyPromise> m_readyPromise;

    bool m_isDocumentLoaded { true };
};

}
