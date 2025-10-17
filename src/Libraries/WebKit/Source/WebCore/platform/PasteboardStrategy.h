/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 27, 2023.
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
#ifndef PasteboardStrategy_h
#define PasteboardStrategy_h

#include <wtf/Forward.h>

namespace WebCore {

class Color;
class SharedBuffer;
class PasteboardContext;
class PasteboardCustomData;
class SelectionData;
class FragmentedSharedBuffer;
struct PasteboardImage;
struct PasteboardItemInfo;
struct PasteboardURL;
struct PasteboardWebContent;

class PasteboardStrategy {
public:
#if PLATFORM(IOS_FAMILY)
    virtual void writeToPasteboard(const PasteboardURL&, const String& pasteboardName, const PasteboardContext*) = 0;
    virtual void writeToPasteboard(const PasteboardWebContent&, const String& pasteboardName, const PasteboardContext*) = 0;
    virtual void writeToPasteboard(const PasteboardImage&, const String& pasteboardName, const PasteboardContext*) = 0;
    virtual void writeToPasteboard(const String& pasteboardType, const String&, const String& pasteboardName, const PasteboardContext*) = 0;
    virtual void updateSupportedTypeIdentifiers(const Vector<String>& identifiers, const String& pasteboardName, const PasteboardContext*) = 0;
#endif // PLATFORM(IOS_FAMILY)
#if PLATFORM(COCOA)
    virtual void getTypes(Vector<String>& types, const String& pasteboardName, const PasteboardContext*) = 0;
    virtual RefPtr<SharedBuffer> bufferForType(const String& pasteboardType, const String& pasteboardName, const PasteboardContext*) = 0;
    virtual void getPathnamesForType(Vector<String>& pathnames, const String& pasteboardType, const String& pasteboardName, const PasteboardContext*) = 0;
    virtual String stringForType(const String& pasteboardType, const String& pasteboardName, const PasteboardContext*) = 0;
    virtual Vector<String> allStringsForType(const String& pasteboardType, const String& pasteboardName, const PasteboardContext*) = 0;
    virtual int64_t changeCount(const String& pasteboardName, const PasteboardContext*) = 0;
    virtual Color color(const String& pasteboardName, const PasteboardContext*) = 0;
    virtual URL url(const String& pasteboardName, const PasteboardContext*) = 0;
    virtual int getNumberOfFiles(const String& pasteboardName, const PasteboardContext*) = 0;

    virtual int64_t addTypes(const Vector<String>& pasteboardTypes, const String& pasteboardName, const PasteboardContext*) = 0;
    virtual int64_t setTypes(const Vector<String>& pasteboardTypes, const String& pasteboardName, const PasteboardContext*) = 0;
    virtual int64_t setBufferForType(SharedBuffer*, const String& pasteboardType, const String& pasteboardName, const PasteboardContext*) = 0;
    virtual int64_t setURL(const PasteboardURL&, const String& pasteboardName, const PasteboardContext*) = 0;
    virtual int64_t setColor(const Color&, const String& pasteboardName, const PasteboardContext*) = 0;
    virtual int64_t setStringForType(const String&, const String& pasteboardType, const String& pasteboardName, const PasteboardContext*) = 0;

    virtual bool containsURLStringSuitableForLoading(const String& pasteboardName, const PasteboardContext*) = 0;
    virtual String urlStringSuitableForLoading(const String& pasteboardName, String& title, const PasteboardContext*) = 0;
#endif
    virtual String readStringFromPasteboard(size_t index, const String& pasteboardType, const String& pasteboardName, const PasteboardContext*) = 0;
    virtual RefPtr<SharedBuffer> readBufferFromPasteboard(std::optional<size_t> index, const String& pasteboardType, const String& pasteboardName, const PasteboardContext*) = 0;
    virtual URL readURLFromPasteboard(size_t index, const String& pasteboardName, String& title, const PasteboardContext*) = 0;
    virtual std::optional<PasteboardItemInfo> informationForItemAtIndex(size_t index, const String& pasteboardName, int64_t changeCount, const PasteboardContext*) = 0;
    virtual std::optional<Vector<PasteboardItemInfo>> allPasteboardItemInfo(const String& pasteboardName, int64_t changeCount, const PasteboardContext*) = 0;
    virtual int getPasteboardItemsCount(const String& pasteboardName, const PasteboardContext*) = 0;

    virtual Vector<String> typesSafeForDOMToReadAndWrite(const String& pasteboardName, const String& origin, const PasteboardContext*) = 0;
    virtual int64_t writeCustomData(const Vector<PasteboardCustomData>&, const String& pasteboardName, const PasteboardContext*) = 0;
    virtual bool containsStringSafeForDOMToReadForType(const String&, const String& pasteboardName, const PasteboardContext*) = 0;

#if PLATFORM(GTK)
    virtual Vector<String> types(const String& pasteboardName) = 0;
    virtual String readTextFromClipboard(const String& pasteboardName) = 0;
    virtual Vector<String> readFilePathsFromClipboard(const String& pasteboardName) = 0;
    virtual RefPtr<SharedBuffer> readBufferFromClipboard(const String& pasteboardName, const String& pasteboardType) = 0;
    virtual void writeToClipboard(const String& pasteboardName, SelectionData&&) = 0;
    virtual void clearClipboard(const String& pasteboardName) = 0;
    virtual int64_t changeCount(const String& pasteboardName) = 0;
#endif // PLATFORM(GTK)

#if USE(LIBWPE)
    virtual void getTypes(Vector<String>& types) = 0;
    virtual void writeToPasteboard(const PasteboardWebContent&) = 0;
    virtual void writeToPasteboard(const String& pasteboardType, const String&) = 0;
#endif

protected:
    virtual ~PasteboardStrategy()
    {
    }
};

}

#endif // !PasteboardStrategy_h
