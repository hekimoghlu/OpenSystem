/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 12, 2023.
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

#include "HTMLElement.h"
#include "Image.h"

namespace WebCore {

enum class AttachmentAssociatedElementType : uint8_t;

class AttachmentAssociatedElement;
class DOMRectReadOnly;
class File;
class HTMLImageElement;
class RenderAttachment;
class ShadowRoot;
class FragmentedSharedBuffer;

class HTMLAttachmentElement final : public HTMLElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(HTMLAttachmentElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(HTMLAttachmentElement);
public:
    static Ref<HTMLAttachmentElement> create(const QualifiedName&, Document&);
    WEBCORE_EXPORT static String getAttachmentIdentifier(HTMLElement&);
    static URL archiveResourceURL(const String&);

    WEBCORE_EXPORT URL blobURL() const;
    WEBCORE_EXPORT File* file() const;

    enum class UpdateDisplayAttributes : bool { No, Yes };
    void setFile(RefPtr<File>&&, UpdateDisplayAttributes = UpdateDisplayAttributes::No);

    const String& uniqueIdentifier() const { return m_uniqueIdentifier; }
    void setUniqueIdentifier(const String&);

    void copyNonAttributePropertiesFromElement(const Element&) final;

    WEBCORE_EXPORT void updateAttributes(std::optional<uint64_t>&& newFileSize, const AtomString& newContentType, const AtomString& newFilename);
    WEBCORE_EXPORT void updateAssociatedElementWithData(const String& contentType, Ref<FragmentedSharedBuffer>&& data);
    WEBCORE_EXPORT void updateThumbnailForNarrowLayout(const RefPtr<Image>& thumbnail);
    WEBCORE_EXPORT void updateThumbnailForWideLayout(Vector<uint8_t>&&);
    WEBCORE_EXPORT void updateIconForNarrowLayout(const RefPtr<Image>& icon, const WebCore::FloatSize&);
    WEBCORE_EXPORT void updateIconForWideLayout(Vector<uint8_t>&&);

    InsertedIntoAncestorResult insertedIntoAncestor(InsertionType, ContainerNode&) final;
    void removedFromAncestor(RemovalType, ContainerNode&) final;

    String ensureUniqueIdentifier();
    AttachmentAssociatedElement* associatedElement() const;
    AttachmentAssociatedElementType associatedElementType() const;

    WEBCORE_EXPORT String attachmentTitle() const;
    const AtomString& attachmentSubtitle() const;
    const AtomString& attachmentActionForDisplay() const;
    String attachmentTitleForDisplay() const;
    const AtomString& attachmentSubtitleForDisplay() const;
    WEBCORE_EXPORT String attachmentType() const;
    String attachmentPath() const;
    RefPtr<Image> thumbnail() const { return m_thumbnail; }
    RefPtr<Image> icon() const { return m_icon; }
    void requestIconIfNeededWithSize(const FloatSize&);
    void requestWideLayoutIconIfNeeded();
    FloatSize iconSize() const { return m_iconSize; }
    void invalidateRendering();
    DOMRectReadOnly* saveButtonClientRect() const;

#if ENABLE(SERVICE_CONTROLS)
    bool isImageMenuEnabled() const { return m_isImageMenuEnabled; }
    void setImageMenuEnabled(bool value) { m_isImageMenuEnabled = value; }
#endif

    bool isWideLayout() const { return m_implementation == Implementation::WideLayout; }
    HTMLElement* wideLayoutShadowContainer() const { return m_containerElement.get(); }
    HTMLElement* wideLayoutImageElement() const;

private:
    friend class AttachmentSaveEventListener;

    HTMLAttachmentElement(const QualifiedName&, Document&);
    virtual ~HTMLAttachmentElement();

    void didAddUserAgentShadowRoot(ShadowRoot&) final;
    void ensureWideLayoutShadowTree(ShadowRoot&);
    void updateProgress(const AtomString&);
    void updateSaveButton(bool);
    void updateImage();

    void setNeedsIconRequest();

    RenderPtr<RenderElement> createElementRenderer(RenderStyle&&, const RenderTreePosition&) final;
    bool isReplaced(const RenderStyle&) const final { return true; }
    bool shouldSelectOnMouseDown() final {
#if PLATFORM(IOS_FAMILY)
        return false;
#else
        return true;
#endif
    }
    bool canContainRangeEndPoint() const final { return false; }
    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) final;

#if ENABLE(SERVICE_CONTROLS)
    bool childShouldCreateRenderer(const Node&) const final;
#endif

    enum class Implementation: uint8_t { NarrowLayout, WideLayout };
    Implementation m_implementation { Implementation::NarrowLayout };

    RefPtr<File> m_file;
    String m_uniqueIdentifier;
    RefPtr<Image> m_thumbnail;
    RefPtr<Image> m_icon;
    FloatSize m_iconSize;

    // The thumbnail is shown if non-empty, otherwise the icon is shown if non-empty.
    Vector<uint8_t> m_thumbnailForWideLayout;
    Vector<uint8_t> m_iconForWideLayout;

    RefPtr<HTMLImageElement> m_imageElement;
    RefPtr<HTMLElement> m_containerElement;
    RefPtr<HTMLElement> m_placeholderElement;
    RefPtr<HTMLElement> m_progressElement;
    RefPtr<HTMLElement> m_informationBlock;
    RefPtr<HTMLElement> m_actionTextElement;
    RefPtr<HTMLElement> m_titleElement;
    RefPtr<HTMLElement> m_subtitleElement;
    RefPtr<HTMLElement> m_saveArea;
    RefPtr<HTMLElement> m_saveButton;
    mutable RefPtr<DOMRectReadOnly> m_saveButtonClientRect;

    bool m_needsIconRequest { true };

#if ENABLE(SERVICE_CONTROLS)
    bool m_isImageMenuEnabled { false };
#endif
};

} // namespace WebCore

#endif // ENABLE(ATTACHMENT_ELEMENT)
