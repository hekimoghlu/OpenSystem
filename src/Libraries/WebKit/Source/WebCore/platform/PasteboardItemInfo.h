/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 22, 2023.
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

#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

enum class PasteboardItemPresentationStyle : uint8_t {
    Unspecified,
    Inline,
    Attachment
};

struct PresentationSize {
    std::optional<double> width;
    std::optional<double> height;
};

struct PasteboardItemInfo {
    Vector<String> pathsForFileUpload;
    Vector<String> platformTypesForFileUpload;
    Vector<String> platformTypesByFidelity;
    String suggestedFileName;
    PresentationSize preferredPresentationSize;
    bool isNonTextType { false };
    bool containsFileURLAndFileUploadContent { false };
    Vector<String> webSafeTypesByFidelity;
    PasteboardItemPresentationStyle preferredPresentationStyle { PasteboardItemPresentationStyle::Unspecified };

    String pathForContentType(const String& type) const
    {
        ASSERT(pathsForFileUpload.size() == platformTypesForFileUpload.size());
        auto index = platformTypesForFileUpload.find(type);
        if (index == notFound)
            return { };

        return pathsForFileUpload[index];
    }

    // The preferredPresentationStyle flag is platform API used by drag or copy sources to explicitly indicate
    // that the data being written to the item provider should be treated as an attachment; unfortunately, not
    // all clients attempt to set this flag, so we additionally take having a suggested filename as a strong
    // indicator that the item should be treated as an attachment or file.
    bool canBeTreatedAsAttachmentOrFile() const
    {
        switch (preferredPresentationStyle) {
        case PasteboardItemPresentationStyle::Inline:
            return false;
        case PasteboardItemPresentationStyle::Attachment:
            return true;
        case PasteboardItemPresentationStyle::Unspecified:
            return !suggestedFileName.isEmpty();
        }
        ASSERT_NOT_REACHED();
        return false;
    }

    String contentTypeForHighestFidelityItem() const
    {
        if (platformTypesForFileUpload.isEmpty())
            return { };

        return platformTypesForFileUpload.first();
    }

    String pathForHighestFidelityItem() const
    {
        if (pathsForFileUpload.isEmpty())
            return { };

        return pathsForFileUpload.first();
    }
};

}
