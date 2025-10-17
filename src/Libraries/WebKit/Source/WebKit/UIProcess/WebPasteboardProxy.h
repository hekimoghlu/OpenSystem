/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 6, 2025.
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

#include "MessageReceiver.h"
#include "SandboxExtension.h"
#include <WebCore/PageIdentifier.h>
#include <WebCore/SharedMemory.h>
#include <wtf/HashMap.h>
#include <wtf/WeakHashSet.h>

namespace IPC {
struct AsyncReplyIDType;
using AsyncReplyID = AtomicObjectIdentifier<AsyncReplyIDType>;
class SharedBufferReference;
}

namespace WebCore {
enum class DataOwnerType : uint8_t;
class Color;
class PasteboardCustomData;
class SelectionData;
struct PasteboardBuffer;
struct PasteboardImage;
struct PasteboardItemInfo;
struct PasteboardURL;
struct PasteboardWebContent;
class SharedBuffer;
}

namespace WebKit {

enum class PasteboardAccessIntent : bool;
class WebFrameProxy;
class WebProcessProxy;

class WebPasteboardProxy : public IPC::MessageReceiver {
    WTF_MAKE_NONCOPYABLE(WebPasteboardProxy);
    friend LazyNeverDestroyed<WebPasteboardProxy>;
public:
    static WebPasteboardProxy& singleton();

    void addWebProcessProxy(WebProcessProxy&);
    void removeWebProcessProxy(WebProcessProxy&);

    // Do nothing since this is a singleton.
    void ref() const final { }
    void deref() const final { }

#if PLATFORM(COCOA)
    void revokeAccess(WebProcessProxy&);
    std::optional<IPC::AsyncReplyID> grantAccessToCurrentData(WebProcessProxy&, const String& pasteboardName, CompletionHandler<void()>&&);
    void grantAccessToCurrentTypes(WebProcessProxy&, const String& pasteboardName);
#endif

#if PLATFORM(GTK)
    void setPrimarySelectionOwner(WebFrameProxy*);
    WebFrameProxy* primarySelectionOwner() const { return m_primarySelectionOwner; }
    void didDestroyFrame(WebFrameProxy*);
#endif

private:
    WebPasteboardProxy();

    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) override;
    bool didReceiveSyncMessage(IPC::Connection&, IPC::Decoder&, UniqueRef<IPC::Encoder>&) override;

    RefPtr<WebProcessProxy> webProcessProxyForConnection(IPC::Connection&) const;

#if PLATFORM(IOS_FAMILY)
    void writeURLToPasteboard(IPC::Connection&, const WebCore::PasteboardURL&, const String& pasteboardName, std::optional<WebCore::PageIdentifier>);
    void writeWebContentToPasteboard(IPC::Connection&, const WebCore::PasteboardWebContent&, const String& pasteboardName, std::optional<WebCore::PageIdentifier>);
    void writeImageToPasteboard(IPC::Connection&, const WebCore::PasteboardImage&, const String& pasteboardName, std::optional<WebCore::PageIdentifier>);
    void writeStringToPasteboard(IPC::Connection&, const String& pasteboardType, const String&, const String& pasteboardName, std::optional<WebCore::PageIdentifier>);
    void updateSupportedTypeIdentifiers(const Vector<String>& identifiers, const String& pasteboardName, std::optional<WebCore::PageIdentifier>);
#endif
#if PLATFORM(COCOA)
    void getNumberOfFiles(IPC::Connection&, const String& pasteboardName, std::optional<WebCore::PageIdentifier>, CompletionHandler<void(uint64_t)>&&);
    void getPasteboardTypes(IPC::Connection&, const String& pasteboardName, std::optional<WebCore::PageIdentifier>, CompletionHandler<void(Vector<String>&&)>&&);
    void getPasteboardPathnamesForType(IPC::Connection&, const String& pasteboardName, const String& pasteboardType, std::optional<WebCore::PageIdentifier>, CompletionHandler<void(Vector<String>&& pathnames, Vector<SandboxExtension::Handle>&&)>&&);
    void getPasteboardStringForType(IPC::Connection&, const String& pasteboardName, const String& pasteboardType, std::optional<WebCore::PageIdentifier>, CompletionHandler<void(String&&)>&&);
    void getPasteboardStringsForType(IPC::Connection&, const String& pasteboardName, const String& pasteboardType, std::optional<WebCore::PageIdentifier>, CompletionHandler<void(Vector<String>&&)>&&);
    void getPasteboardBufferForType(IPC::Connection&, const String& pasteboardName, const String& pasteboardType, std::optional<WebCore::PageIdentifier>, CompletionHandler<void(WebCore::PasteboardBuffer&&)>&&);
    void getPasteboardChangeCount(IPC::Connection&, const String& pasteboardName, std::optional<WebCore::PageIdentifier>, CompletionHandler<void(int64_t)>&&);
    void getPasteboardColor(IPC::Connection&, const String& pasteboardName, std::optional<WebCore::PageIdentifier>, CompletionHandler<void(WebCore::Color&&)>&&);
    void getPasteboardURL(IPC::Connection&, const String& pasteboardName, std::optional<WebCore::PageIdentifier>, CompletionHandler<void(const String&)>&&);
    void addPasteboardTypes(IPC::Connection&, const String& pasteboardName, const Vector<String>& pasteboardTypes, std::optional<WebCore::PageIdentifier>, CompletionHandler<void(int64_t)>&&);
    void setPasteboardTypes(IPC::Connection&, const String& pasteboardName, const Vector<String>& pasteboardTypes, std::optional<WebCore::PageIdentifier>, CompletionHandler<void(int64_t)>&&);
    void setPasteboardURL(IPC::Connection&, const WebCore::PasteboardURL&, const String& pasteboardName, std::optional<WebCore::PageIdentifier>, CompletionHandler<void(int64_t)>&&);
    void setPasteboardColor(IPC::Connection&, const String&, const WebCore::Color&, std::optional<WebCore::PageIdentifier>, CompletionHandler<void(int64_t)>&&);
    void setPasteboardStringForType(IPC::Connection&, const String& pasteboardName, const String& pasteboardType, const String&, std::optional<WebCore::PageIdentifier>, CompletionHandler<void(int64_t)>&&);
    void setPasteboardBufferForType(IPC::Connection&, const String& pasteboardName, const String& pasteboardType, RefPtr<WebCore::SharedBuffer>&&, std::optional<WebCore::PageIdentifier>, CompletionHandler<void(int64_t)>&&);

#if ENABLE(IPC_TESTING_API)
    void testIPCSharedMemory(IPC::Connection&, const String& pasteboardName, const String& pasteboardType, WebCore::SharedMemory::Handle&&, std::optional<WebCore::PageIdentifier>, CompletionHandler<void(int64_t, String)>&&);
#endif

#endif

    void readStringFromPasteboard(IPC::Connection&, size_t index, const String& pasteboardType, const String& pasteboardName, std::optional<WebCore::PageIdentifier>, CompletionHandler<void(String&&)>&&);
    void readURLFromPasteboard(IPC::Connection&, size_t index, const String& pasteboardName, std::optional<WebCore::PageIdentifier>, CompletionHandler<void(String&& url, String&& title)>&&);
    void readBufferFromPasteboard(IPC::Connection&, std::optional<size_t> index, const String& pasteboardType, const String& pasteboardName, std::optional<WebCore::PageIdentifier>, CompletionHandler<void(RefPtr<WebCore::SharedBuffer>&&)>&&);
    void getPasteboardItemsCount(IPC::Connection&, const String& pasteboardName, std::optional<WebCore::PageIdentifier>, CompletionHandler<void(uint64_t)>&&);
    void informationForItemAtIndex(IPC::Connection&, size_t index, const String& pasteboardName, int64_t changeCount, std::optional<WebCore::PageIdentifier>, CompletionHandler<void(std::optional<WebCore::PasteboardItemInfo>&&)>&&);
    void allPasteboardItemInfo(IPC::Connection&, const String& pasteboardName, int64_t changeCount, std::optional<WebCore::PageIdentifier>, CompletionHandler<void(std::optional<Vector<WebCore::PasteboardItemInfo>>&&)>&&);

    void writeCustomData(IPC::Connection&, const Vector<WebCore::PasteboardCustomData>&, const String& pasteboardName, std::optional<WebCore::PageIdentifier>, CompletionHandler<void(int64_t)>&&);
    void typesSafeForDOMToReadAndWrite(IPC::Connection&, const String& pasteboardName, const String& origin, std::optional<WebCore::PageIdentifier>, CompletionHandler<void(Vector<String>&&)>&&);
    void containsStringSafeForDOMToReadForType(IPC::Connection&, const String&, const String& pasteboardName, std::optional<WebCore::PageIdentifier>, CompletionHandler<void(bool)>&&);
    void containsURLStringSuitableForLoading(IPC::Connection&, const String& pasteboardName, std::optional<WebCore::PageIdentifier>, CompletionHandler<void(bool)>&&);
    void urlStringSuitableForLoading(IPC::Connection&, const String& pasteboardName, std::optional<WebCore::PageIdentifier>, CompletionHandler<void(String&& url, String&& title)>&&);

#if PLATFORM(GTK)
    void getTypes(const String& pasteboardName, CompletionHandler<void(Vector<String>&&)>&&);
    void readText(IPC::Connection&, const String& pasteboardName, CompletionHandler<void(String&&)>&&);
    void readFilePaths(IPC::Connection&, const String& pasteboardName, CompletionHandler<void(Vector<String>&&)>&&);
    void readBuffer(IPC::Connection&, const String& pasteboardName, const String& pasteboardType, CompletionHandler<void(RefPtr<WebCore::SharedBuffer>&&)>&&);
    void writeToClipboard(const String& pasteboardName, WebCore::SelectionData&&);
    void clearClipboard(const String& pasteboardName);
    void getPasteboardChangeCount(IPC::Connection&, const String& pasteboardName, CompletionHandler<void(int64_t)>&&);

    WebFrameProxy* m_primarySelectionOwner { nullptr };
#endif // PLATFORM(GTK)

#if USE(LIBWPE)
    void getPasteboardTypes(CompletionHandler<void(Vector<String>&&)>&&);
    void writeWebContentToPasteboard(const WebCore::PasteboardWebContent&);
    void writeStringToPasteboard(const String& pasteboardType, const String&);
#endif

#if PLATFORM(COCOA)
    bool canAccessPasteboardTypes(IPC::Connection&, const String& pasteboardName) const;
    bool canAccessPasteboardData(IPC::Connection&, const String& pasteboardName) const;
    void didModifyContentsOfPasteboard(IPC::Connection&, const String& pasteboardName, int64_t previousChangeCount, int64_t newChangeCount);

    enum class PasteboardAccessType : uint8_t { Types, TypesAndData };
    std::optional<PasteboardAccessType> accessType(IPC::Connection&, const String& pasteboardName) const;
    void grantAccess(WebProcessProxy&, const String& pasteboardName, PasteboardAccessType);

    std::optional<WebCore::DataOwnerType> determineDataOwner(IPC::Connection&, const String& pasteboardName, std::optional<WebCore::PageIdentifier>, PasteboardAccessIntent) const;
#endif

    WeakHashSet<WebProcessProxy> m_webProcessProxySet;

#if PLATFORM(COCOA)
    struct PasteboardAccessInformation {
        ~PasteboardAccessInformation();

        int64_t changeCount { 0 };
        Vector<std::pair<WeakPtr<WebProcessProxy>, PasteboardAccessType>> processes;

        void grantAccess(WebProcessProxy&, PasteboardAccessType);
        void revokeAccess(WebProcessProxy&);
        std::optional<PasteboardAccessType> accessType(WebProcessProxy&) const;
    };
    HashMap<String, PasteboardAccessInformation> m_pasteboardNameToAccessInformationMap;
#endif
};

} // namespace WebKit
