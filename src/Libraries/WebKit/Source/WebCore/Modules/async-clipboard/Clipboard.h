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

#include "EventTarget.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class ClipboardItem;
class DeferredPromise;
class LocalFrame;
class Navigator;
class Pasteboard;
class PasteboardCustomData;

class Clipboard final : public RefCounted<Clipboard>, public EventTarget {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(Clipboard);
public:
    static Ref<Clipboard> create(Navigator&);
    ~Clipboard();

    enum EventTargetInterfaceType eventTargetInterface() const final;
    ScriptExecutionContext* scriptExecutionContext() const final;

    LocalFrame* frame() const;
    Navigator* navigator();

    using RefCounted::ref;
    using RefCounted::deref;

    void readText(Ref<DeferredPromise>&&);
    void writeText(const String& data, Ref<DeferredPromise>&&);

    void read(Ref<DeferredPromise>&&);
    void write(const Vector<Ref<ClipboardItem>>& data, Ref<DeferredPromise>&&);

    void getType(ClipboardItem&, const String& type, Ref<DeferredPromise>&&);

private:
    Clipboard(Navigator&);

    struct Session {
        std::unique_ptr<Pasteboard> pasteboard;
        Vector<Ref<ClipboardItem>> items;
        int64_t changeCount;
    };

    void refEventTarget() final { ref(); }
    void derefEventTarget() final { deref(); }

    Pasteboard& activePasteboard();

    enum class SessionIsValid : bool { No, Yes };
    SessionIsValid updateSessionValidity();

    class ItemWriter : public RefCounted<ItemWriter> {
    public:
        static Ref<ItemWriter> create(Clipboard& clipboard, Ref<DeferredPromise>&& promise)
        {
            return adoptRef(*new ItemWriter(clipboard, WTFMove(promise)));
        }

        ~ItemWriter();

        void write(const Vector<Ref<ClipboardItem>>&);
        void invalidate();

    private:
        ItemWriter(Clipboard&, Ref<DeferredPromise>&&);

        void setData(std::optional<PasteboardCustomData>&&, size_t index);
        void didSetAllData();
        void reject();

        WeakPtr<Clipboard, WeakPtrImplWithEventTargetData> m_clipboard;
        Vector<std::optional<PasteboardCustomData>> m_dataToWrite;
        RefPtr<DeferredPromise> m_promise;
        unsigned m_pendingItemCount;
        std::unique_ptr<Pasteboard> m_pasteboard;
        int64_t m_changeCountAtStart { 0 };
    };

    void didResolveOrReject(ItemWriter&);

    std::optional<Session> m_activeSession;
    WeakPtr<Navigator> m_navigator;
    Vector<std::optional<PasteboardCustomData>> m_dataToWrite;
    RefPtr<ItemWriter> m_activeItemWriter;
};

} // namespace WebCore
