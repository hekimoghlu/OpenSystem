/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 19, 2022.
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
#include "DictationCommandIOS.h"

#if PLATFORM(IOS_FAMILY)

#include "Document.h"
#include "DocumentMarkerController.h"
#include "Element.h"
#include "Position.h"
#include "SmartReplace.h"
#include "TextIterator.h"
#include "VisibleUnits.h"

namespace WebCore {

DictationCommandIOS::DictationCommandIOS(Ref<Document>&& document, Vector<Vector<String>>&& dictationPhrases, id metadata)
    : CompositeEditCommand(WTFMove(document), EditAction::Dictation)
    , m_dictationPhrases(WTFMove(dictationPhrases))
    , m_metadata(metadata)
{
}

Ref<DictationCommandIOS> DictationCommandIOS::create(Ref<Document>&& document, Vector<Vector<String>>&& dictationPhrases, id metadata)
{
    return adoptRef(*new DictationCommandIOS(WTFMove(document), WTFMove(dictationPhrases), metadata));
}

void DictationCommandIOS::doApply()
{
    uint64_t resultLength = 0;
    for (auto& interpretations : m_dictationPhrases) {
        const String& firstInterpretation = interpretations[0];
        resultLength += firstInterpretation.length();
        inputText(firstInterpretation, true);

        if (interpretations.size() > 1) {
            auto alternatives = interpretations;
            alternatives.remove(0);
            addMarker(*endingSelection().toNormalizedRange(), DocumentMarkerType::DictationPhraseWithAlternatives, WTFMove(alternatives));
        }

        setEndingSelection(VisibleSelection(endingSelection().visibleEnd()));
    }

    // FIXME: Add the result marker using a Position cached before results are inserted, instead of relying on character counts.

    auto endPosition = endingSelection().visibleEnd();
    auto end = makeBoundaryPoint(endPosition);
    auto* root = endPosition.rootEditableElement();
    if (!end || !root)
        return;

    auto endOffset = characterCount({ { *root, 0 }, WTFMove(*end) });
    if (endOffset < resultLength)
        return;

    auto resultRange = resolveCharacterRange(makeRangeSelectingNodeContents(*root), { endOffset - resultLength, endOffset });
    addMarker(resultRange, DocumentMarkerType::DictationResult, m_metadata);
}

} // namespace WebCore

#endif // PLATFORM(IOS_FAMILY)
