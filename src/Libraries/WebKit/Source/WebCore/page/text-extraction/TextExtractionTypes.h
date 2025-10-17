/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 20, 2023.
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

#include "CharacterRange.h"
#include "FloatRect.h"
#include "FloatSize.h"
#include <wtf/Forward.h>
#include <wtf/URL.h>

namespace WebCore {
namespace TextExtraction {

struct Editable {
    String label;
    String placeholder;
    bool isSecure { false };
    bool isFocused { false };
};

struct TextItemData {
    Vector<std::pair<URL, CharacterRange>> links;
    std::optional<CharacterRange> selectedRange;
    String content;
    std::optional<Editable> editable;
};

struct ScrollableItemData {
    FloatSize contentSize;
};

struct ImageItemData {
    String name;
    String altText;
};

enum class ContainerType : uint8_t {
    Root,
    ViewportConstrained,
    List,
    ListItem,
    BlockQuote,
    Article,
    Section,
    Nav,
    Button,
};

using ItemData = std::variant<ContainerType, TextItemData, ScrollableItemData, ImageItemData>;

struct Item {
    ItemData data;
    FloatRect rectInRootView;
    Vector<Item> children;
};

} // namespace TextExtraction
} // namespace WebCore
