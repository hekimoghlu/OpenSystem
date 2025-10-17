/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 2, 2025.
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

#include "YarrFlags.h"
#include <wtf/OptionSet.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/text/WTFString.h>

namespace JSC { namespace Yarr {

class JS_EXPORT_PRIVATE RegularExpression {
    WTF_MAKE_TZONE_ALLOCATED(RegularExpression);
public:
    explicit RegularExpression(StringView, OptionSet<Flags> = { });
    ~RegularExpression();

    RegularExpression(const RegularExpression&);
    RegularExpression& operator=(const RegularExpression&);

    int match(StringView, unsigned startFrom = 0, int* matchLength = nullptr) const;
    int searchRev(StringView) const;

    int matchedLength() const;
    bool isValid() const;

private:
    class Private;
    RefPtr<Private> d;
};

void JS_EXPORT_PRIVATE replace(String&, const RegularExpression&, StringView);

} } // namespace JSC::Yarr
