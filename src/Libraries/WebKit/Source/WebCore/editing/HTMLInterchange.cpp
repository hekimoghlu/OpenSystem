/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 20, 2022.
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
#include "HTMLInterchange.h"

#include "Editing.h"
#include "RenderStyleInlines.h"
#include "RenderText.h"
#include "Text.h"
#include <wtf/text/MakeString.h>
#include <wtf/text/StringBuilder.h>
#include <wtf/unicode/CharacterNames.h>

namespace WebCore {

String convertHTMLTextToInterchangeFormat(const String& in, const Text* node)
{
    // Assume all the text comes from node.
    if (node->renderer() && node->renderer()->style().preserveNewline())
        return in;

    static NeverDestroyed<const String> convertedSpaceString { makeString("<span class=\""_s, AppleConvertedSpace, "\">"_s, noBreakSpace, "</span>"_s) };

    StringBuilder s;

    unsigned i = 0;
    unsigned consumed = 0;
    while (i < in.length()) {
        consumed = 1;
        if (deprecatedIsCollapsibleWhitespace(in[i])) {
            // count number of adjoining spaces
            unsigned j = i + 1;
            while (j < in.length() && deprecatedIsCollapsibleWhitespace(in[j]))
                j++;
            unsigned count = j - i;
            consumed = count;
            while (count) {
                unsigned add = count % 3;
                switch (add) {
                    case 0:
                        s.append(convertedSpaceString.get(), ' ', convertedSpaceString.get());
                        add = 3;
                        break;
                    case 1:
                        if (i == 0 || i + 1 == in.length()) // at start or end of string
                            s.append(convertedSpaceString.get());
                        else
                            s.append(' ');
                        break;
                    case 2:
                        if (i == 0) {
                             // at start of string
                            s.append(convertedSpaceString.get(), ' ');
                        } else if (i + 2 == in.length()) {
                             // at end of string
                            s.append(convertedSpaceString.get(), convertedSpaceString.get());
                        } else {
                            s.append(convertedSpaceString.get(), ' ');
                        }
                        break;
                }
                count -= add;
            }
        } else
            s.append(in[i]);
        i += consumed;
    }

    return s.toString();
}

} // namespace WebCore
