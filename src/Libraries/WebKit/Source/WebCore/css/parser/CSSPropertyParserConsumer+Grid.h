/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 12, 2022.
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

namespace WebCore {

class CSSGridLineNamesValue;
class CSSParserTokenRange;
class CSSValue;

struct CSSParserContext;
struct NamedGridAreaMap;

enum CSSValueID : uint16_t;

namespace CSSPropertyParserHelpers {

// https://drafts.csswg.org/css-grid/

enum class AllowEmpty : bool { No, Yes };
enum TrackListType : uint8_t { GridTemplate, GridTemplateNoRepeat, GridAuto };

bool isGridBreadthIdent(CSSValueID);
bool parseGridTemplateAreasRow(StringView gridRowNames, NamedGridAreaMap&, const size_t rowCount, size_t& columnCount);
RefPtr<CSSGridLineNamesValue> consumeGridLineNames(CSSParserTokenRange&, const CSSParserContext&, AllowEmpty = AllowEmpty::No);
RefPtr<CSSValue> consumeGridLine(CSSParserTokenRange&, const CSSParserContext&);
RefPtr<CSSValue> consumeGridTrackSize(CSSParserTokenRange&, const CSSParserContext&);
RefPtr<CSSValue> consumeGridTrackList(CSSParserTokenRange&, const CSSParserContext&, TrackListType);
RefPtr<CSSValue> consumeGridTemplatesRowsOrColumns(CSSParserTokenRange&, const CSSParserContext&);
RefPtr<CSSValue> consumeGridTemplateAreas(CSSParserTokenRange&, const CSSParserContext&);
RefPtr<CSSValue> consumeGridAutoFlow(CSSParserTokenRange&, const CSSParserContext&);
RefPtr<CSSValue> consumeMasonryAutoFlow(CSSParserTokenRange&, const CSSParserContext&);

} // namespace CSSPropertyParserHelpers
} // namespace WebCore
