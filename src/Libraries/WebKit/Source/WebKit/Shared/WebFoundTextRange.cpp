/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 13, 2023.
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
#include "WebFoundTextRange.h"

#include <wtf/StdLibExtras.h>

namespace WebKit {

unsigned WebFoundTextRange::hash() const
{
    return WTF::switchOn(data,
        [] (const WebFoundTextRange::DOMData& domData) {
            return pairIntHash(domData.location, domData.length);
        },
        [] (const WebFoundTextRange::PDFData& pdfData) {
            return pairIntHash(pairIntHash(pairIntHash(pdfData.startPage, pdfData.endPage), pdfData.startOffset), pdfData.endOffset);
        }
    );
}

bool WebFoundTextRange::operator==(const WebFoundTextRange& other) const
{
    if (frameIdentifier.isHashTableDeletedValue())
        return other.frameIdentifier.isHashTableDeletedValue();
    if (other.frameIdentifier.isHashTableDeletedValue())
        return false;

    return data == other.data
        && frameIdentifier == other.frameIdentifier
        && order == other.order;
}

} // namespace WebKit
