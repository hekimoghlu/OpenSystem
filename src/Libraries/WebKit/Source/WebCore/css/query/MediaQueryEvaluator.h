/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 30, 2025.
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

#include "GenericMediaQueryEvaluator.h"
#include "MediaQuery.h"

namespace WebCore {

class RenderStyle;

namespace MQ {

class MediaQueryEvaluator : public GenericMediaQueryEvaluator<MediaQueryEvaluator> {
public:
    MediaQueryEvaluator(const AtomString& mediaType, const Document&, const RenderStyle* rootElementStyle);
    MediaQueryEvaluator(const AtomString& mediaType = nullAtom(), EvaluationResult mediaConditionResult = EvaluationResult::False);

    bool evaluate(const MediaQueryList&) const;
    bool evaluate(const MediaQuery&) const;

    bool evaluateMediaType(const MediaQuery&) const;

    OptionSet<MediaQueryDynamicDependency> collectDynamicDependencies(const MediaQueryList&) const;
    OptionSet<MediaQueryDynamicDependency> collectDynamicDependencies(const MediaQuery&) const;

    bool isPrintMedia() const;

private:
    AtomString m_mediaType;
    WeakPtr<const Document, WeakPtrImplWithEventTargetData> m_document;
    const RenderStyle* m_rootElementStyle { nullptr }; // FIXME: Switch to a smart pointer.
    EvaluationResult m_staticMediaConditionResult { EvaluationResult::Unknown };
};

}
}
