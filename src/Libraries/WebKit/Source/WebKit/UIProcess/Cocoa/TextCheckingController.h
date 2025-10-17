/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 25, 2023.
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

#if ENABLE(PLATFORM_DRIVEN_TEXT_CHECKING)

#import "EditingRange.h"
#import <WebCore/FloatRect.h>
#import <wtf/CompletionHandler.h>
#import <wtf/TZoneMalloc.h>

OBJC_CLASS NSAttributedString;

namespace WebKit {

class WebPageProxy;

class TextCheckingController final {
    WTF_MAKE_TZONE_ALLOCATED(TextCheckingController);
    WTF_MAKE_NONCOPYABLE(TextCheckingController);
public:
    explicit TextCheckingController(WebPageProxy&);
    ~TextCheckingController() = default;

    void replaceRelativeToSelection(NSAttributedString *annotatedString, int64_t selectionOffset, uint64_t length, uint64_t relativeReplacementLocation, uint64_t relativeReplacementLength);
    void removeAnnotationRelativeToSelection(NSString *annotationName, int64_t selectionOffset, uint64_t length);

private:
    WeakRef<WebPageProxy> m_page;
};

} // namespace WebKit

#endif // ENABLE(PLATFORM_DRIVEN_TEXT_CHECKING)
