/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 18, 2023.
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

#include "RegistrableDomain.h"
#include "Supplementable.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
class DocumentStorageAccess;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::DocumentStorageAccess> : std::true_type { };
}

namespace WebCore {

class DeferredPromise;
class Document;
class UserGestureIndicator;
class WeakPtrImplWithEventTargetData;

enum class StorageAccessWasGranted : uint8_t { No, Yes, YesWithException };

enum class StorageAccessPromptWasShown : bool { No, Yes };

enum class StorageAccessScope : bool {
    PerFrame,
    PerPage
};

enum class StorageAccessQuickResult : bool {
    Grant,
    Reject
};

struct RequestStorageAccessResult {
    StorageAccessWasGranted wasGranted;
    StorageAccessPromptWasShown promptWasShown;
    StorageAccessScope scope;
    RegistrableDomain topFrameDomain;
    RegistrableDomain subFrameDomain;
};

const unsigned maxNumberOfTimesExplicitlyDeniedStorageAccess = 2;

class DocumentStorageAccess final : public Supplement<Document>, public CanMakeWeakPtr<DocumentStorageAccess> {
    WTF_MAKE_TZONE_ALLOCATED(DocumentStorageAccess);
public:
    explicit DocumentStorageAccess(Document&);
    ~DocumentStorageAccess();

    static void hasStorageAccess(Document&, Ref<DeferredPromise>&&);
    static bool hasStorageAccessForDocumentQuirk(Document&);

    static void requestStorageAccess(Document&, Ref<DeferredPromise>&&);
    static void requestStorageAccessForDocumentQuirk(Document&, CompletionHandler<void(StorageAccessWasGranted)>&&);
    static void requestStorageAccessForNonDocumentQuirk(Document& hostingDocument, RegistrableDomain&& requestingDomain, CompletionHandler<void(StorageAccessWasGranted)>&&);

private:
    std::optional<bool> hasStorageAccessQuickCheck();
    void hasStorageAccess(Ref<DeferredPromise>&&);
    bool hasStorageAccessQuirk();

    std::optional<StorageAccessQuickResult> requestStorageAccessQuickCheck();
    void requestStorageAccess(Ref<DeferredPromise>&&);
    void requestStorageAccessForDocumentQuirk(CompletionHandler<void(StorageAccessWasGranted)>&&);
    void requestStorageAccessForNonDocumentQuirk(RegistrableDomain&& requestingDomain, CompletionHandler<void(StorageAccessWasGranted)>&&);
    void requestStorageAccessQuirk(RegistrableDomain&& requestingDomain, CompletionHandler<void(StorageAccessWasGranted)>&&);

    static DocumentStorageAccess* from(Document&);
    static ASCIILiteral supplementName();
    bool hasFrameSpecificStorageAccess() const;
    void setWasExplicitlyDeniedFrameSpecificStorageAccess() { ++m_numberOfTimesExplicitlyDeniedFrameSpecificStorageAccess; };
    bool isAllowedToRequestStorageAccess() { return m_numberOfTimesExplicitlyDeniedFrameSpecificStorageAccess < maxNumberOfTimesExplicitlyDeniedStorageAccess; };
    void enableTemporaryTimeUserGesture();
    void consumeTemporaryTimeUserGesture();

    Ref<Document> protectedDocument() const;

    std::unique_ptr<UserGestureIndicator> m_temporaryUserGesture;
    
    WeakRef<Document, WeakPtrImplWithEventTargetData> m_document;

    uint8_t m_numberOfTimesExplicitlyDeniedFrameSpecificStorageAccess = 0;

    StorageAccessScope m_storageAccessScope = StorageAccessScope::PerPage;
};

} // namespace WebCore
