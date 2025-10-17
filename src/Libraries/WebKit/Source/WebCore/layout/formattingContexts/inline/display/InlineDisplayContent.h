/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 8, 2023.
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

#include "InlineDisplayBox.h"
#include "InlineDisplayLine.h"
#include <wtf/Vector.h>

namespace WebCore {
namespace InlineDisplay {

using Boxes = Vector<Box>;
using Lines = Vector<Line>;

struct Content {
    WTF_MAKE_STRUCT_FAST_ALLOCATED;

    void clear();
    void set(Content&& newContent);
    void append(Content&& newContent);
    void insert(Content&& newContent, size_t lineIndex, size_t boxIndex);
    void remove(size_t firstLineIndex, size_t numberOfLines, size_t firstBoxIndex, size_t numberOfBoxes);

    Lines lines;
    Boxes boxes;
};

}
}
