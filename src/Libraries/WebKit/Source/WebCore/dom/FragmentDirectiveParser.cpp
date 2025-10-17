/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 14, 2025.
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
#include "FragmentDirectiveParser.h"

#include "FragmentDirectiveUtilities.h"
#include "Logging.h"
#include <wtf/Deque.h>
#include <wtf/URL.h>
#include <wtf/URLParser.h>
#include <wtf/text/TextStream.h>

namespace WebCore {

FragmentDirectiveParser::FragmentDirectiveParser(StringView fragmentDirective)
{
    parseFragmentDirective(fragmentDirective);
    
    m_fragmentDirective = fragmentDirective;
    m_isValid = true;
}

FragmentDirectiveParser::~FragmentDirectiveParser() = default;

// https://wicg.github.io/scroll-to-text-fragment/#parse-a-text-directive
void FragmentDirectiveParser::parseFragmentDirective(StringView fragmentDirective)
{
    LOG_WITH_STREAM(TextFragment, stream << " parseFragmentDirective: ");
    
    Vector<ParsedTextDirective> parsedTextDirectives;
    String textDirectivePrefix = "text="_s;

    auto directives = fragmentDirective.split('&');
    
    LOG_WITH_STREAM(TextFragment, stream << " parseFragmentDirective: ");
    
    for (auto directive : directives) {
        if (!directive.startsWith(textDirectivePrefix))
            continue;
        
        auto textDirective = directive.substring(textDirectivePrefix.length());
        
        Deque<String> tokens;
        bool containsEmptyToken = false;
        for (auto token : textDirective.split(',')) {
            if (token.isEmpty()) {
                LOG_WITH_STREAM(TextFragment, stream << " empty token ");
                containsEmptyToken = true;
                break;
            }
            tokens.append(token.toString());
        }
        if (containsEmptyToken)
            continue;
        if (tokens.size() > 4 || tokens.size() < 1) {
            LOG_WITH_STREAM(TextFragment, stream << " wrong number of tokens ");
            continue;
        }
        
        ParsedTextDirective parsedTextDirective;
        
        if (tokens.first().endsWith('-') && tokens.first().length() > 1) {
            auto takenFirstToken = tokens.takeFirst();
            if (auto prefix = WTF::URLParser::formURLDecode(StringView(takenFirstToken).left(takenFirstToken.length() - 1)))
                parsedTextDirective.prefix = WTFMove(*prefix);
            else
                LOG_WITH_STREAM(TextFragment, stream << " could not decode prefix ");
        }
        
        if (tokens.isEmpty()) {
            LOG_WITH_STREAM(TextFragment, stream << " not enough tokens ");
            continue;
        }

        if (tokens.last().startsWith('-') && tokens.last().length() > 1) {
            tokens.last() = tokens.last().substring(1);
            
            if (auto suffix = WTF::URLParser::formURLDecode(tokens.takeLast()))
                parsedTextDirective.suffix = WTFMove(*suffix);
            else
                LOG_WITH_STREAM(TextFragment, stream << " could not decode suffix ");
        }
        
        if (tokens.size() != 1 && tokens.size() != 2) {
            LOG_WITH_STREAM(TextFragment, stream << " not enough tokens ");
            continue;
        }
        
        if (auto start = WTF::URLParser::formURLDecode(tokens.first()))
            parsedTextDirective.startText = WTFMove(*start);
        else
            LOG_WITH_STREAM(TextFragment, stream << " could not decode start ");
        
        if (tokens.size() == 2) {
            if (auto end = WTF::URLParser::formURLDecode(tokens.last()))
                parsedTextDirective.endText = WTFMove(*end);
            else
                LOG_WITH_STREAM(TextFragment, stream << " could not decode end ");
        }
        
        parsedTextDirectives.append(parsedTextDirective);
    }
    
    m_parsedTextDirectives = parsedTextDirectives;
}

} // namespace WebCore
