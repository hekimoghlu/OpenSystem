/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 15, 2021.
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
#include "ComposedTreeIterator.h"

#include <wtf/text/TextStream.h>

namespace WebCore {

String composedTreeAsText(ContainerNode& root, ComposedTreeAsTextMode mode)
{
    TextStream stream;
    auto descendants = composedTreeDescendants(root);
    for (auto it = descendants.begin(), end = descendants.end(); it != end; ++it) {
        writeIndent(stream, it.depth());

        if (is<Text>(*it)) {
            stream << "#text";
            if (mode == ComposedTreeAsTextMode::WithPointers)
                stream << " " << &*it;
            stream << "\n";
            continue;
        }
        auto& element = downcast<Element>(*it);
        stream << element.localName();
        if (element.shadowRoot())
            stream << " (shadow root)";
        if (mode == ComposedTreeAsTextMode::WithPointers)
            stream << " " << &*it;
        stream << "\n";
    }
    return stream.release();
}

}
