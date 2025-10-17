/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 31, 2023.
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

#include "DocumentFragment.h"
#include "LocalFrame.h"
#include "Pasteboard.h"
#include "SimpleRange.h"
#include "markup.h"
#include <wtf/WeakRef.h>

namespace WebCore {

class ArchiveResource;

class FrameWebContentReader : public PasteboardWebContentReader {
public:
    FrameWebContentReader(LocalFrame& frame)
        : m_frame(frame)
    {
    }

    LocalFrame& frame() const { return m_frame.get(); }
    Ref<LocalFrame> protectedFrame() const { return m_frame.get(); }

protected:
    bool shouldSanitize() const;
    MSOListQuirks msoListQuirksForMarkup() const;

private:
    WeakRef<LocalFrame> m_frame;
};

class WebContentReader final : public FrameWebContentReader {
public:
#if PLATFORM(COCOA)
    static constexpr auto placeholderAttachmentFilenamePrefix = "webkit-attachment-"_s;
#endif

    WebContentReader(LocalFrame& frame, const SimpleRange& context, bool allowPlainText)
        : FrameWebContentReader(frame)
        , m_context(context)
#if PLATFORM(COCOA) || PLATFORM(GTK)
        , m_allowPlainText(allowPlainText)
#endif
    {
        UNUSED_PARAM(allowPlainText);
    }

    void addFragment(Ref<DocumentFragment>&&);
    RefPtr<DocumentFragment> takeFragment() { return std::exchange(m_fragment, nullptr); }
    RefPtr<DocumentFragment> protectedFragment() const { return m_fragment; }

    bool madeFragmentFromPlainText() const { return m_madeFragmentFromPlainText; }

private:
#if PLATFORM(COCOA) || PLATFORM(GTK)
    bool readFilePath(const String&, PresentationSize preferredPresentationSize = { }, const String& contentType = { }) override;
    bool readFilePaths(const Vector<String>&) override;
    bool readHTML(const String&) override;
    bool readImage(Ref<FragmentedSharedBuffer>&&, const String& type, PresentationSize preferredPresentationSize = { }) override;
    bool readURL(const URL&, const String& title) override;
    bool readPlainText(const String&) override;
#endif

#if PLATFORM(COCOA)
    bool readWebArchive(SharedBuffer&) override;
    bool readRTFD(SharedBuffer&) override;
    bool readRTF(SharedBuffer&) override;
    bool readDataBuffer(SharedBuffer&, const String& type, const AtomString& name, PresentationSize preferredPresentationSize = { }) override;
#endif

    const SimpleRange m_context;
#if PLATFORM(COCOA) || PLATFORM(GTK)
    const bool m_allowPlainText;
#endif

    RefPtr<DocumentFragment> m_fragment;
    bool m_madeFragmentFromPlainText { false };
};

class WebContentMarkupReader final : public FrameWebContentReader {
public:
    explicit WebContentMarkupReader(LocalFrame& frame)
        : FrameWebContentReader(frame)
    {
    }

    String takeMarkup() { return std::exchange(m_markup, { }); }

private:
#if PLATFORM(COCOA) || PLATFORM(GTK)
    bool readFilePath(const String&, PresentationSize = { }, const String& = { }) override { return false; }
    bool readFilePaths(const Vector<String>&) override { return false; }
    bool readHTML(const String&) override;
    bool readImage(Ref<FragmentedSharedBuffer>&&, const String&, PresentationSize = { }) override { return false; }
    bool readURL(const URL&, const String&) override { return false; }
    bool readPlainText(const String&) override { return false; }
#endif

#if PLATFORM(COCOA)
    bool readWebArchive(SharedBuffer&) override;
    bool readRTFD(SharedBuffer&) override;
    bool readRTF(SharedBuffer&) override;
    bool readDataBuffer(SharedBuffer&, const String&, const AtomString&, PresentationSize = { }) override { return false; }
#endif

    String m_markup;
};

#if PLATFORM(COCOA) && defined(__OBJC__)
struct FragmentAndResources {
    RefPtr<DocumentFragment> fragment;
    Vector<Ref<ArchiveResource>> resources;
};

enum class FragmentCreationOptions : uint8_t {
    IgnoreResources = 1 << 0,
    NoInterchangeNewlines = 1 << 1,
    SanitizeMarkup = 1 << 2
};

WEBCORE_EXPORT RefPtr<DocumentFragment> createFragment(LocalFrame&, NSAttributedString *, OptionSet<FragmentCreationOptions> = { });
#endif

}
