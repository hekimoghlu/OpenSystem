/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 20, 2023.
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

#include "DataOwnerType.h"
#include <wtf/Forward.h>
#include <wtf/Function.h>
#include <wtf/RetainPtr.h>
#include <wtf/Vector.h>

#if PLATFORM(MAC)
OBJC_CLASS NSPasteboard;
OBJC_CLASS NSPasteboardItem;
#endif

#if PLATFORM(IOS_FAMILY)
OBJC_CLASS UIPasteboard;
OBJC_PROTOCOL(AbstractPasteboard);
#endif

#if USE(LIBWPE)
struct wpe_pasteboard;
#endif

namespace WebCore {

class Color;
class SharedBuffer;
class PasteboardCustomData;
class SelectionData;
struct PasteboardBuffer;
struct PasteboardImage;
struct PasteboardItemInfo;
struct PasteboardURL;
struct PasteboardWebContent;
class FragmentedSharedBuffer;

class PlatformPasteboard {
public:
    WEBCORE_EXPORT explicit PlatformPasteboard(const String& pasteboardName);
#if PLATFORM(IOS_FAMILY) || USE(LIBWPE)
    WEBCORE_EXPORT PlatformPasteboard();
    WEBCORE_EXPORT void updateSupportedTypeIdentifiers(const Vector<String>& types);
#endif
    WEBCORE_EXPORT std::optional<PasteboardItemInfo> informationForItemAtIndex(size_t index, int64_t changeCount);
    WEBCORE_EXPORT std::optional<Vector<PasteboardItemInfo>> allPasteboardItemInfo(int64_t changeCount);

    WEBCORE_EXPORT static void performAsDataOwner(DataOwnerType, NOESCAPE Function<void()>&&);

    enum class IncludeImageTypes : bool { No, Yes };
    static String platformPasteboardTypeForSafeTypeForDOMToReadAndWrite(const String& domType, IncludeImageTypes = IncludeImageTypes::No);

    WEBCORE_EXPORT void getTypes(Vector<String>& types) const;
    WEBCORE_EXPORT PasteboardBuffer bufferForType(const String& pasteboardType) const;
    WEBCORE_EXPORT void getPathnamesForType(Vector<String>& pathnames, const String& pasteboardType) const;
    WEBCORE_EXPORT String stringForType(const String& pasteboardType) const;
    WEBCORE_EXPORT Vector<String> allStringsForType(const String& pasteboardType) const;
    WEBCORE_EXPORT int64_t changeCount() const;
    WEBCORE_EXPORT Color color();
    WEBCORE_EXPORT URL url();

    // Take ownership of the pasteboard, and return new change count.
    WEBCORE_EXPORT int64_t addTypes(const Vector<String>& pasteboardTypes);
    WEBCORE_EXPORT int64_t setTypes(const Vector<String>& pasteboardTypes);

    // These methods will return 0 if pasteboard ownership has been taken from us.
    WEBCORE_EXPORT int64_t copy(const String& fromPasteboard);
    WEBCORE_EXPORT int64_t setBufferForType(SharedBuffer*, const String& pasteboardType);
    WEBCORE_EXPORT int64_t setURL(const PasteboardURL&);
    WEBCORE_EXPORT int64_t setColor(const Color&);
    WEBCORE_EXPORT int64_t setStringForType(const String&, const String& pasteboardType);
    WEBCORE_EXPORT void write(const PasteboardWebContent&);
    WEBCORE_EXPORT void write(const PasteboardImage&);
    WEBCORE_EXPORT void write(const String& pasteboardType, const String&);
    WEBCORE_EXPORT void write(const PasteboardURL&);
    WEBCORE_EXPORT RefPtr<SharedBuffer> readBuffer(std::optional<size_t> index, const String& pasteboardType) const;
    WEBCORE_EXPORT String readString(size_t index, const String& pasteboardType) const;
    WEBCORE_EXPORT URL readURL(size_t index, String& title) const;
    WEBCORE_EXPORT int count() const;
    WEBCORE_EXPORT int numberOfFiles() const;
    WEBCORE_EXPORT int64_t write(const Vector<PasteboardCustomData>&);
    WEBCORE_EXPORT int64_t write(const PasteboardCustomData&);
    WEBCORE_EXPORT Vector<String> typesSafeForDOMToReadAndWrite(const String& origin) const;
    WEBCORE_EXPORT bool containsStringSafeForDOMToReadForType(const String&) const;

#if PLATFORM(COCOA)
    WEBCORE_EXPORT bool containsURLStringSuitableForLoading();
    WEBCORE_EXPORT String urlStringSuitableForLoading(String& title);
#endif

private:
#if PLATFORM(IOS_FAMILY)
    bool allowReadingURLAtIndex(const URL&, int index) const;
#endif

#if PLATFORM(MAC)
    NSPasteboardItem *itemAtIndex(size_t index) const;
#endif

#if PLATFORM(MAC)
    RetainPtr<NSPasteboard> m_pasteboard;
#endif
#if PLATFORM(IOS_FAMILY)
    RetainPtr<AbstractPasteboard> m_pasteboard;
#endif
#if USE(LIBWPE)
    struct wpe_pasteboard* m_pasteboard;
#endif
};

} // namespace WebCore
