/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 6, 2022.
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

#include "TextExtractionTypes.h"
#include <wtf/Expected.h>

namespace WebCore {

class Element;
class FloatRect;
class LocalFrame;
class Page;
enum class ExceptionCode : uint8_t;

namespace TextExtraction {

WEBCORE_EXPORT Item extractItem(std::optional<WebCore::FloatRect>&& collectionRectInRootView, Page&);
WEBCORE_EXPORT Vector<std::pair<String, FloatRect>> extractAllTextAndRects(Page&);

struct RenderedText {
    String textWithReplacedContent;
    String textWithoutReplacedContent;
    bool hasLargeReplacedDescendant { false };
};

RenderedText extractRenderedText(Element&);

} // namespace TextExtraction
} // namespace WebCore
