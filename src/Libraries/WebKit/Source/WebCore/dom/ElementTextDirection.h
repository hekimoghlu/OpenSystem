/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 14, 2023.
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

#include <optional>

namespace WTF {
class AtomString;
};

namespace WebCore {

class Element;

enum class TextDirection : bool;

// https://html.spec.whatwg.org/multipage/dom.html#attr-dir
enum class TextDirectionState : uint8_t {
    LTR,
    RTL,
    Auto,
    Undefined,
};

TextDirectionState parseTextDirectionState(const WTF::AtomString&);
TextDirectionState elementTextDirectionState(const Element&);

bool elementHasValidTextDirectionState(const Element&);
bool elementHasAutoTextDirectionState(const Element&);

std::optional<TextDirection> computeAutoDirectionality(const Element&);
std::optional<TextDirection> computeTextDirectionIfDirIsAuto(const Element&);

void textDirectionStateChanged(Element&, TextDirectionState);

void updateEffectiveTextDirectionState(Element&, TextDirectionState, Element* initiator = nullptr);
void updateEffectiveTextDirectionOfDescendants(Element&, std::optional<TextDirection>, Element* initiator = nullptr);
void updateEffectiveTextDirectionOfAncestors(Element&, Element* initiator);

} // namespace WebCore
