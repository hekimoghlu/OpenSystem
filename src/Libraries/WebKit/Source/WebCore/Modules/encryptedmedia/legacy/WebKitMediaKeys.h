/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 10, 2022.
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

#if ENABLE(LEGACY_ENCRYPTED_MEDIA)

#include "ExceptionOr.h"
#include "LegacyCDM.h"
#include <JavaScriptCore/Forward.h>
#include <wtf/Vector.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class Document;
class WeakPtrImplWithEventTargetData;
class HTMLMediaElement;
class WebKitMediaKeySession;

class WebKitMediaKeys final : public RefCounted<WebKitMediaKeys>, private LegacyCDMClient {
    WTF_MAKE_FAST_ALLOCATED;
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(WebKitMediaKeys);
public:
    static ExceptionOr<Ref<WebKitMediaKeys>> create(const String& keySystem);
    virtual ~WebKitMediaKeys();

    ExceptionOr<Ref<WebKitMediaKeySession>> createSession(Document&, const String& mimeType, Ref<Uint8Array>&& initData);
    static bool isTypeSupported(const String& keySystem, const String& mimeType);
    const String& keySystem() const { return m_keySystem; }

    LegacyCDM& cdm() { return m_cdm; }

    void setMediaElement(HTMLMediaElement*);

    void keyAdded();
    RefPtr<ArrayBuffer> cachedKeyForKeyId(const String& keyId) const;

private:
    RefPtr<MediaPlayer> cdmMediaPlayer(const LegacyCDM*) const final;

    WebKitMediaKeys(const String& keySystem, Ref<LegacyCDM>&&);

    Vector<Ref<WebKitMediaKeySession>> m_sessions;
    WeakPtr<HTMLMediaElement> m_mediaElement;
    String m_keySystem;
    Ref<LegacyCDM> m_cdm;
};

}

#endif // ENABLE(LEGACY_ENCRYPTED_MEDIA)
