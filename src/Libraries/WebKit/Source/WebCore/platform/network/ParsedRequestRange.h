/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 14, 2021.
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

#include <wtf/Forward.h>
#include <wtf/HashTraits.h>
#include <wtf/text/StringView.h>

namespace WebCore {

struct ParsedRequestRange {
public:
    WEBCORE_EXPORT static std::optional<ParsedRequestRange> parse(StringView);
    static std::optional<ParsedRequestRange> parse(const String& string) { return parse(StringView(string)); }

    const size_t begin { 0 };
    const size_t end { 0 };

private:
    ParsedRequestRange(size_t begin, size_t end)
        : begin(begin)
        , end(end) { }
};

}
