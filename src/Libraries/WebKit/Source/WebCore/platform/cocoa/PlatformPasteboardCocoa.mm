/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 26, 2023.
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
#import "config.h"
#import "PlatformPasteboard.h"

#import "Pasteboard.h"
#import "PasteboardItemInfo.h"
#import "WebCoreNSURLExtras.h"
#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>

#if PLATFORM(IOS_FAMILY)
#import "AbstractPasteboard.h"
#else
#import "LegacyNSPasteboardTypes.h"
#endif

namespace WebCore {

std::optional<Vector<PasteboardItemInfo>> PlatformPasteboard::allPasteboardItemInfo(int64_t changeCount)
{
    if (changeCount != [m_pasteboard changeCount])
        return std::nullopt;

    Vector<PasteboardItemInfo> itemInfo;
    int numberOfItems = count();
    itemInfo.reserveInitialCapacity(numberOfItems);
    for (NSInteger itemIndex = 0; itemIndex < numberOfItems; ++itemIndex) {
        auto item = informationForItemAtIndex(itemIndex, changeCount);
        if (!item)
            return std::nullopt;

        itemInfo.append(WTFMove(*item));
    }
    return itemInfo;
}

bool PlatformPasteboard::containsStringSafeForDOMToReadForType(const String& type) const
{
    return !stringForType(type).isEmpty();
}

String PlatformPasteboard::urlStringSuitableForLoading(String& title)
{
#if PLATFORM(MAC)
    String URLTitleString = stringForType(String(WebURLNamePboardType));
    if (!URLTitleString.isEmpty())
        title = URLTitleString;
#endif

    Vector<String> types;
    getTypes(types);

#if PLATFORM(IOS_FAMILY)
    UNUSED_PARAM(title);
    String urlPasteboardType = UTTypeURL.identifier;
    String stringPasteboardType = UTTypeText.identifier;
#else
    String urlPasteboardType = legacyURLPasteboardType();
    String stringPasteboardType = legacyStringPasteboardType();
#endif

    if (types.contains(urlPasteboardType)) {
        NSURL *URLFromPasteboard = [NSURL URLWithString:stringForType(urlPasteboardType)];
        // Cannot drop other schemes unless <rdar://problem/10562662> and <rdar://problem/11187315> are fixed.
        if (URL { URLFromPasteboard }.protocolIsInHTTPFamily())
            return [URLByCanonicalizingURL(URLFromPasteboard) absoluteString];
    }

    if (types.contains(stringPasteboardType)) {
        NSURL *URLFromPasteboard = [NSURL URLWithString:stringForType(stringPasteboardType)];
        // Pasteboard content is not trusted, because JavaScript code can modify it. We can sanitize it for URLs and other typed content, but not for strings.
        // The result of this function is used to initiate navigation, so we shouldn't allow arbitrary file URLs.
        // FIXME: Should we allow only http family schemes, or anything non-local?
        if (URL { URLFromPasteboard }.protocolIsInHTTPFamily())
            return [URLByCanonicalizingURL(URLFromPasteboard) absoluteString];
    }

#if PLATFORM(MAC)
    if (types.contains(String(legacyFilenamesPasteboardType()))) {
        Vector<String> files;
        getPathnamesForType(files, String(legacyFilenamesPasteboardType()));
        if (files.size() == 1) {
            BOOL isDirectory;
            if ([[NSFileManager defaultManager] fileExistsAtPath:files[0] isDirectory:&isDirectory] && isDirectory)
                return String();
            return [URLByCanonicalizingURL([NSURL fileURLWithPath:files[0]]) absoluteString];
        }
    }
#endif

    return { };
}

} // namespace WebCore
