/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 22, 2023.
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
#include "TextEncodingDetector.h"

#include "TextEncoding.h"
#include <unicode/ucnv.h>
#include <unicode/ucsdet.h>
#include <wtf/text/icu/UnicodeExtras.h>

namespace PAL {

bool detectTextEncoding(std::span<const uint8_t> data, ASCIILiteral hintEncodingName, TextEncoding* detectedEncoding)
{
    *detectedEncoding = TextEncoding();
    UErrorCode status = U_ZERO_ERROR;
    UCharsetDetector* detector = ucsdet_open(&status);
    if (U_FAILURE(status))
        return false;
    ucsdet_enableInputFilter(detector, true);
    ucsdet_setText(detector, byteCast<char>(data.data()), static_cast<int32_t>(data.size()), &status);
    if (U_FAILURE(status))
        return false;

    // FIXME: A few things we can do other than improving
    // the ICU detector itself. 
    // 1. Use ucsdet_detectAll and pick the most likely one given
    // "the context" (parent-encoding, referrer encoding, etc).
    // 2. 'Emulate' Firefox/IE's non-Universal detectors (e.g.
    // Chinese, Japanese, Russian, Korean and Hebrew) by picking the 
    // encoding with a highest confidence among the detector-specific
    // limited set of candidate encodings.
    // Below is a partial implementation of the first part of what's outlined
    // above.
    auto matches = ucsdet_detectAll_span(detector, &status);
    if (U_FAILURE(status)) {
        ucsdet_close(detector);
        return false;
    }

    const char* encoding = nullptr;
    if (!hintEncodingName.isNull()) {
        TextEncoding hintEncoding(hintEncodingName);
        // 10 is the minimum confidence value consistent with the codepoint
        // allocation in a given encoding. The size of a chunk passed to
        // us varies even for the same html file (apparently depending on 
        // the network load). When we're given a rather short chunk, we 
        // don't have a sufficiently reliable signal other than the fact that
        // the chunk is consistent with a set of encodings. So, instead of
        // setting an arbitrary threshold, we have to scan all the encodings
        // consistent with the data.  
        const int32_t kThreshold = 10;
        for (auto* match : matches) {
            int32_t confidence = ucsdet_getConfidence(match, &status);
            if (U_FAILURE(status)) {
                status = U_ZERO_ERROR;
                continue;
            }
            if (confidence < kThreshold)
                break;
            const char* matchEncoding = ucsdet_getName(match, &status);
            if (U_FAILURE(status)) {
                status = U_ZERO_ERROR;
                continue;
            }
            if (TextEncoding(StringView::fromLatin1(matchEncoding)) == hintEncoding) {
                encoding = hintEncodingName;
                break;
            }
        }
    }
    // If no match is found so far, just pick the top match. 
    // This can happen, say, when a parent frame in EUC-JP refers to
    // a child frame in Shift_JIS and both frames do NOT specify the encoding
    // making us resort to auto-detection (when it IS turned on).
    if (!encoding && !matches.empty())
        encoding = ucsdet_getName(matches[0], &status);
    if (U_SUCCESS(status)) {
        *detectedEncoding = TextEncoding(StringView::fromLatin1(encoding));
        ucsdet_close(detector);
        return true;
    }    
    ucsdet_close(detector);
    return false;
}

}
