/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 5, 2025.
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
#include "CSSFontFace.h"
#include "CSSPropertyNames.h"
#include "ExceptionOr.h"
#include "IDLTypes.h"
#include <variant>
#include <wtf/UniqueRef.h>
#include <wtf/WeakPtr.h>

namespace JSC {
class ArrayBuffer;
class ArrayBufferView;
}

namespace WebCore {

template<typename IDLType> class DOMPromiseProxyWithResolveCallback;

class FontFace final : public RefCounted<FontFace>, public ActiveDOMObject, public CSSFontFaceClient {
public:
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    struct Descriptors {
        String style;
        String weight;
        String width;
        String unicodeRange;
        String featureSettings;
        String display;
        String sizeAdjust;
    };

    using RefCounted::ref;
    using RefCounted::deref;
    
    using Source = std::variant<String, RefPtr<JSC::ArrayBuffer>, RefPtr<JSC::ArrayBufferView>>;
    static Ref<FontFace> create(ScriptExecutionContext&, const String& family, Source&&, const Descriptors&);
    static Ref<FontFace> create(ScriptExecutionContext*, CSSFontFace&);
    virtual ~FontFace();

    ExceptionOr<void> setFamily(ScriptExecutionContext&, const String&);
    ExceptionOr<void> setStyle(ScriptExecutionContext&, const String&);
    ExceptionOr<void> setWeight(ScriptExecutionContext&, const String&);
    ExceptionOr<void> setWidth(ScriptExecutionContext&, const String&);
    ExceptionOr<void> setUnicodeRange(ScriptExecutionContext&, const String&);
    ExceptionOr<void> setFeatureSettings(ScriptExecutionContext&, const String&);
    ExceptionOr<void> setDisplay(ScriptExecutionContext&, const String&);
    ExceptionOr<void> setSizeAdjust(ScriptExecutionContext&, const String&);

    String family() const;
    String style() const;
    String weight() const;
    String width() const;
    String unicodeRange() const;
    String featureSettings() const;
    String display() const;
    String sizeAdjust() const;

    enum class LoadStatus { Unloaded, Loading, Loaded, Error };
    LoadStatus status() const;

    using LoadedPromise = DOMPromiseProxyWithResolveCallback<IDLInterface<FontFace>>;
    LoadedPromise& loadedForBindings();
    LoadedPromise& loadForBindings();

    void adopt(CSSFontFace&);

    CSSFontFace& backing() { return m_backing; }

    void fontStateChanged(CSSFontFace&, CSSFontFace::Status oldState, CSSFontFace::Status newState) final;

private:
    explicit FontFace(CSSFontSelector&);
    explicit FontFace(ScriptExecutionContext*, CSSFontFace&);

    // ActiveDOMObject.
    bool virtualHasPendingActivity() const final;

    // Callback for LoadedPromise.
    FontFace& loadedPromiseResolve();
    void setErrorState();

    Ref<CSSFontFace> m_backing;
    UniqueRef<LoadedPromise> m_loadedPromise;
    bool m_mayLoadedPromiseBeScriptObservable { false };
};

}
