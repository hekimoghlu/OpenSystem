/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 26, 2022.
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
#include "InlineDisplayContent.h"

namespace WebCore {
namespace InlineDisplay {

void Content::clear()
{
    lines.clear();
    boxes.clear();
}

void Content::set(Content&& newContent)
{
    lines = WTFMove(newContent.lines);
    boxes = WTFMove(newContent.boxes);
}

void Content::append(Content&& newContent)
{
    lines.appendVector(WTFMove(newContent.lines));
    boxes.appendVector(WTFMove(newContent.boxes));
}

void Content::insert(Content&& newContent, size_t lineIndex, size_t boxIndex)
{
    lines.insertVector(lineIndex, WTFMove(newContent.lines));
    boxes.insertVector(boxIndex, WTFMove(newContent.boxes));
}

void Content::remove(size_t firstLineIndex, size_t numberOfLines, size_t firstBoxIndex, size_t numberOfBoxes)
{
    lines.remove(firstLineIndex, numberOfLines);
    boxes.remove(firstBoxIndex, numberOfBoxes);
}

}
}

