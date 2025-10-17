/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 27, 2023.
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

#include "InspectorProtocolObjects.h"
#include <wtf/Forward.h>

namespace JSC { namespace Yarr {
class RegularExpression;
} }

namespace Inspector {

namespace ContentSearchUtilities {

enum class SearchType { Regex, ExactString, ContainsString };
enum class SearchCaseSensitive { No, Yes };

using Searcher = std::variant<String, JSC::Yarr::RegularExpression>;
JS_EXPORT_PRIVATE Searcher createSearcherForString(const String&, SearchType, SearchCaseSensitive);
JS_EXPORT_PRIVATE bool searcherMatchesText(const Searcher&, const String& text);

JS_EXPORT_PRIVATE JSC::Yarr::RegularExpression createRegularExpressionForString(const String&, SearchType, SearchCaseSensitive);

JS_EXPORT_PRIVATE int countRegularExpressionMatches(const JSC::Yarr::RegularExpression&, const String&);
JS_EXPORT_PRIVATE Ref<JSON::ArrayOf<Protocol::GenericTypes::SearchMatch>> searchInTextByLines(const String& text, const String& query, const bool caseSensitive, const bool isRegex);
JS_EXPORT_PRIVATE TextPosition textPositionFromOffset(size_t offset, const Vector<size_t>& lineEndings);
JS_EXPORT_PRIVATE Vector<size_t> lineEndings(const String&);

JS_EXPORT_PRIVATE String findStylesheetSourceMapURL(const String& content);

} // namespace ContentSearchUtilities

} // namespace Inspector
