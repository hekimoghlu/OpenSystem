/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 6, 2023.
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

#if PLATFORM(COCOA)

#include "Image.h"
#include <wtf/RefPtr.h>
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

struct TextAttachmentMissingImage {
};

struct TextAttachmentFileWrapper {
#if !PLATFORM(IOS_FAMILY)
    bool ignoresOrientation = false;
#endif
    String preferredFilename;
    RetainPtr<CFDataRef> data;
    String accessibilityLabel;
};

#if ENABLE(MULTI_REPRESENTATION_HEIC)

struct MultiRepresentationHEICAttachmentSingleImage {
    RefPtr<Image> image;
    FloatSize size;
};

struct MultiRepresentationHEICAttachmentData {
    String identifier;
    String description;
    String credit;
    String digitalSourceType;
    Vector<MultiRepresentationHEICAttachmentSingleImage> images;

    // Not serialized.
    // FIXME: Remove this once same-process AttributedString to NSAttributeedString conversion
    // is removed. See https://bugs.webkit.org/show_bug.cgi?id=269384.
    RetainPtr<CFDataRef> data { };
};

#endif

} // namespace WebCore

#endif // PLATFORM(COCOA)
