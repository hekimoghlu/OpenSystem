/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 25, 2024.
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
#include "config.h"
#include "InlineIteratorLogicalOrderTraversal.h"

#include "InlineIteratorLineBox.h"
#include <algorithm>

namespace WebCore {
namespace InlineIterator {

static TextLogicalOrderCache makeTextLogicalOrderCacheIfNeeded(const RenderText& text)
{
    if (!text.needsVisualReordering())
        return { };

    auto cache = makeUnique<TextLogicalOrderCacheData>();
    for (auto textBox : textBoxesFor(text))
        cache->boxes.append(textBox);

    if (cache->boxes.isEmpty())
        return nullptr;

    std::sort(cache->boxes.begin(), cache->boxes.end(), [&](auto& a, auto& b) {
        return a->start() < b->start();
    });

    return cache;
}

static void updateTextLogicalOrderCacheIfNeeded(const TextBoxIterator& textBox, TextLogicalOrderCache& cache)
{
    if (!cache && !(cache = makeTextLogicalOrderCacheIfNeeded(textBox->renderer())))
        return;

    if (cache->index < cache->boxes.size() && cache->boxes[cache->index] == textBox)
        return;

    cache->index = cache->boxes.find(textBox);

    if (cache->index == notFound) {
        cache = { };
        updateTextLogicalOrderCacheIfNeeded(textBox, cache);
    }
}

std::pair<TextBoxIterator, TextLogicalOrderCache> firstTextBoxInLogicalOrderFor(const RenderText& text)
{
    if (auto cache = makeTextLogicalOrderCacheIfNeeded(text))
        return { cache->boxes.first(), WTFMove(cache) };

    return { lineLeftmostTextBoxFor(text), nullptr };
}

TextBoxIterator nextTextBoxInLogicalOrder(const TextBoxIterator& textBox, TextLogicalOrderCache& cache)
{
    updateTextLogicalOrderCacheIfNeeded(textBox, cache);

    if (!cache)
        return textBox->nextTextBox();

    cache->index++;

    if (cache->index < cache->boxes.size())
        return cache->boxes[cache->index];

    return { };
}

static LineLogicalOrderCache makeLineLogicalOrderCache(const LineBoxIterator& lineBox)
{
    auto cache = makeUnique<LineLogicalOrderCacheData>();

    cache->lineBox = lineBox;
    cache->boxes = leafBoxesInLogicalOrder(lineBox, [](auto span) {
        std::ranges::reverse(span);
    });

    return cache;
}

static void updateLineLogicalOrderCacheIfNeeded(const LeafBoxIterator& box, LineLogicalOrderCache& cache)
{
    auto lineBox = box->lineBox();
    if (!cache || cache->lineBox != lineBox)
        cache = makeLineLogicalOrderCache(lineBox);

    if (cache->index < cache->boxes.size() && cache->boxes[cache->index] == box)
        return;

    cache->index = cache->boxes.find(box);

    ASSERT(cache->index != notFound);
}

LeafBoxIterator firstLeafOnLineInLogicalOrder(const LineBoxIterator& lineBox, LineLogicalOrderCache& cache)
{
    cache = makeLineLogicalOrderCache(lineBox);

    if (cache->boxes.isEmpty())
        return { };

    cache->index = 0;
    return cache->boxes.first();
}

LeafBoxIterator lastLeafOnLineInLogicalOrder(const LineBoxIterator& lineBox, LineLogicalOrderCache& cache)
{
    cache = makeLineLogicalOrderCache(lineBox);

    if (cache->boxes.isEmpty())
        return { };

    cache->index = cache->boxes.size() - 1;
    return cache->boxes.last();
}

LeafBoxIterator nextLeafOnLineInLogicalOrder(const LeafBoxIterator& box, LineLogicalOrderCache& cache)
{
    updateLineLogicalOrderCacheIfNeeded(box, cache);

    cache->index++;

    if (cache->index < cache->boxes.size())
        return cache->boxes[cache->index];

    return { };
}

LeafBoxIterator previousLeafOnLineInLogicalOrder(const LeafBoxIterator& box, LineLogicalOrderCache& cache)
{
    updateLineLogicalOrderCacheIfNeeded(box, cache);

    if (!cache->index)
        return { };

    cache->index--;

    return cache->boxes[cache->index];
}

LeafBoxIterator firstLeafOnLineInLogicalOrderWithNode(const LineBoxIterator& lineBox, LineLogicalOrderCache& cache)
{
    auto box = firstLeafOnLineInLogicalOrder(lineBox, cache);
    while (box && !box->renderer().node())
        box = nextLeafOnLineInLogicalOrder(box, cache);
    return box;
}

LeafBoxIterator lastLeafOnLineInLogicalOrderWithNode(const LineBoxIterator& lineBox, LineLogicalOrderCache& cache)
{
    auto box = lastLeafOnLineInLogicalOrder(lineBox, cache);
    while (box && !box->renderer().node())
        box = previousLeafOnLineInLogicalOrder(box, cache);
    return box;
}

}
}
