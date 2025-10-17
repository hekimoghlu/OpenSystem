/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 7, 2024.
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

#include "ExceptionOr.h"
#include <wtf/KeyValuePair.h>
#include <wtf/Ref.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/Vector.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class Blob;
class Clipboard;
class ClipboardItemDataSource;
class DeferredPromise;
class DOMPromise;
class WeakPtrImplWithEventTargetData;
class Navigator;
class PasteboardCustomData;
class ScriptExecutionContext;
struct PasteboardItemInfo;

class ClipboardItem : public RefCountedAndCanMakeWeakPtr<ClipboardItem> {
public:
    ~ClipboardItem();

    enum class PresentationStyle : uint8_t { Unspecified, Inline, Attachment };

    struct Options {
        PresentationStyle presentationStyle { PresentationStyle::Unspecified };
    };

    static ExceptionOr<Ref<ClipboardItem>> create(Vector<KeyValuePair<String, Ref<DOMPromise>>>&&, const Options&);
    static Ref<ClipboardItem> create(Clipboard&, const PasteboardItemInfo&);
    static Ref<Blob> blobFromString(ScriptExecutionContext*, const String& stringData, const String& type);

    Vector<String> types() const;
    void getType(const String&, Ref<DeferredPromise>&&);
    static bool supports(const String& type);

    void collectDataForWriting(Clipboard& destination, CompletionHandler<void(std::optional<PasteboardCustomData>)>&&);

    PresentationStyle presentationStyle() const { return m_presentationStyle; };
    Navigator* navigator();
    Clipboard* clipboard();

private:
    ClipboardItem(Vector<KeyValuePair<String, Ref<DOMPromise>>>&&, const Options&);
    ClipboardItem(Clipboard&, const PasteboardItemInfo&);

    WeakPtr<Clipboard, WeakPtrImplWithEventTargetData> m_clipboard;
    WeakPtr<Navigator> m_navigator;
    std::unique_ptr<ClipboardItemDataSource> m_dataSource;
    PresentationStyle m_presentationStyle { PresentationStyle::Unspecified };
};

} // namespace WebCore
