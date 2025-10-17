/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 15, 2024.
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
#include "TextNodeTraversal.h"

#include "ContainerNode.h"
#include <wtf/text/StringBuilder.h>

namespace WebCore {
namespace TextNodeTraversal {

void appendContents(const ContainerNode& root, StringBuilder& result)
{
    for (Text* text = TextNodeTraversal::firstWithin(root); text; text = TextNodeTraversal::next(*text, &root))
        result.append(text->data());
}

String contentsAsString(const ContainerNode& root)
{
    StringBuilder result;
    appendContents(root, result);
    return result.toString();
}

String contentsAsString(const Node& root)
{
    if (auto text = dynamicDowncast<Text>(root))
        return text->data();
    if (auto containerNode = dynamicDowncast<ContainerNode>(root))
        return contentsAsString(*containerNode);
    return String();
}

String childTextContent(const ContainerNode& root)
{
    StringBuilder result;
    for (Text* text = TextNodeTraversal::firstChild(root); text; text = TextNodeTraversal::nextSibling(*text))
        result.append(text->data());
    return result.toString();
}

}
}
