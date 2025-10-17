/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 16, 2024.
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

#include "ClipboardItemDataSource.h"
#include "ExceptionCode.h"
#include "FileReaderLoaderClient.h"
#include <variant>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class Blob;
class SharedBuffer;
class DOMPromise;
class WeakPtrImplWithEventTargetData;
class FileReaderLoader;
class PasteboardCustomData;
class ScriptExecutionContext;

class ClipboardItemBindingsDataSource : public ClipboardItemDataSource {
    WTF_MAKE_TZONE_ALLOCATED(ClipboardItemBindingsDataSource);
public:
    ClipboardItemBindingsDataSource(ClipboardItem&, Vector<KeyValuePair<String, Ref<DOMPromise>>>&&);
    ~ClipboardItemBindingsDataSource();

private:
    Vector<String> types() const final;
    void getType(const String&, Ref<DeferredPromise>&&) final;
    void collectDataForWriting(Clipboard& destination, CompletionHandler<void(std::optional<PasteboardCustomData>)>&&) final;

    void clearItemTypeLoaders();
    void invokeCompletionHandler();

    using BufferOrString = std::variant<String, Ref<SharedBuffer>>;
    class ClipboardItemTypeLoader : public FileReaderLoaderClient, public RefCounted<ClipboardItemTypeLoader> {
    public:
        static Ref<ClipboardItemTypeLoader> create(Clipboard& writingDestination, const String& type, CompletionHandler<void()>&& completionHandler)
        {
            return adoptRef(*new ClipboardItemTypeLoader(writingDestination, type, WTFMove(completionHandler)));
        }

        ~ClipboardItemTypeLoader();

        void didResolveToString(const String&);
        void didFailToResolve();
        void didResolveToBlob(ScriptExecutionContext&, Ref<Blob>&&);

        void invokeCompletionHandler();

        const String& type() { return m_type; }
        const BufferOrString& data() { return m_data; }

    private:
        ClipboardItemTypeLoader(Clipboard&, const String& type, CompletionHandler<void()>&&);

        void sanitizeDataIfNeeded();

        String dataAsString() const;

        // FileReaderLoaderClient methods.
        void didStartLoading() final { }
        void didReceiveData() final { }
        void didFinishLoading() final;
        void didFail(ExceptionCode) final;

        String m_type;
        BufferOrString m_data;
        std::unique_ptr<FileReaderLoader> m_blobLoader;
        CompletionHandler<void()> m_completionHandler;
        WeakPtr<Clipboard, WeakPtrImplWithEventTargetData> m_writingDestination;
    };

    unsigned m_numberOfPendingClipboardTypes { 0 };
    CompletionHandler<void(std::optional<PasteboardCustomData>)> m_completionHandler;
    Vector<Ref<ClipboardItemTypeLoader>> m_itemTypeLoaders;
    WeakPtr<Clipboard, WeakPtrImplWithEventTargetData> m_writingDestination;

    Vector<KeyValuePair<String, Ref<DOMPromise>>> m_itemPromises;
};

} // namespace WebCore
