/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 10, 2022.
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

#include "ExceptionOr.h"
#include <wtf/Forward.h>
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>

namespace WebCore {

class Pattern;
class SourceImage;
struct DOMMatrix2DInit;

class CanvasPattern : public RefCounted<CanvasPattern> {
public:
    static Ref<CanvasPattern> create(SourceImage&&, bool repeatX, bool repeatY, bool originClean);
    ~CanvasPattern();

    static bool parseRepetitionType(const String&, bool& repeatX, bool& repeatY);

    Pattern& pattern() { return m_pattern; }
    const Pattern& pattern() const { return m_pattern; }

    bool originClean() const { return m_originClean; }
    
    ExceptionOr<void> setTransform(DOMMatrix2DInit&&);

private:
    CanvasPattern(SourceImage&&, bool repeatX, bool repeatY, bool originClean);

    Ref<Pattern> m_pattern;
    bool m_originClean;
};

} // namespace WebCore
