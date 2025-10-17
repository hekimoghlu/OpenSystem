/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 26, 2025.
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

#if ENABLE(ATTACHMENT_ELEMENT)

#include "APIObject.h"
#include "WKBase.h"
#include <WebCore/AttachmentAssociatedElement.h>
#include <wtf/Function.h>
#include <wtf/Lock.h>
#include <wtf/RefPtr.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/WTFString.h>

OBJC_CLASS NSData;
OBJC_CLASS NSFileWrapper;
OBJC_CLASS NSString;

namespace WebCore {
enum class AttachmentAssociatedElementType : uint8_t;

class SharedBuffer;
class FragmentedSharedBuffer;
}

namespace WebKit {
class WebPageProxy;
}

namespace API {

class Attachment final : public ObjectImpl<Object::Type::Attachment> {
public:
    static Ref<Attachment> create(const WTF::String& identifier, WebKit::WebPageProxy&);
    virtual ~Attachment();

    enum class InsertionState : uint8_t { NotInserted, Inserted };

    const WTF::String& identifier() const { return m_identifier; }
    void updateAttributes(CompletionHandler<void()>&&);

    void invalidate();
    bool isValid() const { return !!m_webPage; }

#if PLATFORM(COCOA)
    void cloneFileWrapperTo(Attachment&);
    bool shouldUseFileWrapperIconForDirectory() const;
    void doWithFileWrapper(Function<void(NSFileWrapper *)>&&) const;
    void setFileWrapper(NSFileWrapper *);
    void setFileWrapperAndUpdateContentType(NSFileWrapper *, NSString *contentType);
    WTF::String utiType() const;
#endif
    WTF::String mimeType() const;

    const WTF::String& filePath() const { return m_filePath; }
    void setFilePath(const WTF::String& filePath) { m_filePath = filePath; }
    WTF::String fileName() const;

    const WTF::String& contentType() const { return m_contentType; }
    void setContentType(const WTF::String& contentType) { m_contentType = contentType; }

    InsertionState insertionState() const { return m_insertionState; }
    void setInsertionState(InsertionState state) { m_insertionState = state; }

    bool isEmpty() const;

    RefPtr<WebCore::FragmentedSharedBuffer> associatedElementData() const;
#if PLATFORM(COCOA)
    NSData *associatedElementNSData() const;
#endif
    std::optional<uint64_t> fileSizeForDisplay() const;

    void setAssociatedElementType(WebCore::AttachmentAssociatedElementType associatedElementType) { m_associatedElementType = associatedElementType; }
    WebCore::AttachmentAssociatedElementType associatedElementType() const { return m_associatedElementType; }

    RefPtr<WebCore::SharedBuffer> createSerializedRepresentation() const;
    void updateFromSerializedRepresentation(Ref<WebCore::SharedBuffer>&&, const WTF::String& contentType);

private:
    explicit Attachment(const WTF::String& identifier, WebKit::WebPageProxy&);

#if PLATFORM(COCOA)
    mutable Lock m_fileWrapperLock;
    RetainPtr<NSFileWrapper> m_fileWrapper WTF_GUARDED_BY_LOCK(m_fileWrapperLock);
#endif
    WTF::String m_identifier;
    WTF::String m_filePath;
    WTF::String m_contentType;
    WeakPtr<WebKit::WebPageProxy> m_webPage;
    InsertionState m_insertionState { InsertionState::NotInserted };
    WebCore::AttachmentAssociatedElementType m_associatedElementType { WebCore::AttachmentAssociatedElementType::None };
    bool m_hasEnclosingImage { false };
    bool m_isCreatedFromSerializedRepresentation { false };
};

} // namespace API

#endif // ENABLE(ATTACHMENT_ELEMENT)
