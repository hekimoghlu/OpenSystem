/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 18, 2023.
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
#include "ServerTimingParser.h"

#include "HeaderFieldTokenizer.h"
#include "ServerTiming.h"

#include <wtf/text/CString.h>

namespace WebCore {

namespace ServerTimingParser {

Vector<ServerTiming> parseServerTiming(const String& headerValue)
{
    auto entries = Vector<ServerTiming>();
    if (headerValue.isNull())
        return entries;

    ASSERT(headerValue.is8Bit());

    HeaderFieldTokenizer tokenizer(headerValue);
    while (!tokenizer.isConsumed()) {
        String name = tokenizer.consumeToken();
        if (name.isNull())
            break;

        ServerTiming entry(WTFMove(name));

        while (tokenizer.consume(';')) {
            String parameterName = tokenizer.consumeToken();
            if (parameterName.isNull())
                break;

            String value = emptyString();
            if (tokenizer.consume('=')) {
                value = tokenizer.consumeTokenOrQuotedString();
                tokenizer.consumeBeforeAnyCharMatch({',', ';'});
            }
            entry.setParameter(parameterName, value);
        }

        entries.append(WTFMove(entry));

        if (!tokenizer.consume(','))
            break;
    }
    return entries;
}

} // namespace ServerTimingParser

} // namespace WebCore
