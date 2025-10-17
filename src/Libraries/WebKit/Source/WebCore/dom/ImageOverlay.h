/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 9, 2025.
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

#include "IntRect.h"
#include <wtf/RefCounted.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class Document;
class FloatRect;
class HTMLElement;
class Node;
class SharedBuffer;
class VisibleSelection;

struct CharacterRange;
struct SimpleRange;
struct TextRecognitionResult;

namespace ImageOverlay {

WEBCORE_EXPORT bool hasOverlay(const HTMLElement&);
WEBCORE_EXPORT bool isDataDetectorResult(const HTMLElement&);
WEBCORE_EXPORT bool isInsideOverlay(const SimpleRange&);
WEBCORE_EXPORT bool isInsideOverlay(const Node&);
WEBCORE_EXPORT bool isOverlayText(const Node&);
WEBCORE_EXPORT bool isOverlayText(const Node*);
void removeOverlaySoonIfNeeded(HTMLElement&);
IntRect containerRect(HTMLElement&);
std::optional<CharacterRange> characterRange(const VisibleSelection&);
bool isInsideOverlay(const VisibleSelection&);

#if ENABLE(IMAGE_ANALYSIS)
enum class CacheTextRecognitionResults : bool { No, Yes };
WEBCORE_EXPORT void updateWithTextRecognitionResult(HTMLElement&, const TextRecognitionResult&, CacheTextRecognitionResults = CacheTextRecognitionResults::Yes);
#endif

} // namespace ImageOverlay

} // namespace WebCore
