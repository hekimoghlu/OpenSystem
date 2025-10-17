/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 6, 2023.
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

#include "FloatRect.h"
#include "InlineIteratorInlineBox.h"
#include "InlineIteratorSVGTextBox.h"
#include "SVGTextFragment.h"
#include <wtf/Vector.h>

namespace WebCore {

class LegacyInlineFlowBox;
class RenderObject;
class SVGInlineTextBox;

class SVGTextQuery {
public:
    SVGTextQuery(RenderObject*);

    unsigned numberOfCharacters() const;
    float textLength() const;
    float subStringLength(unsigned startPosition, unsigned length) const;
    FloatPoint startPositionOfCharacter(unsigned position) const;
    FloatPoint endPositionOfCharacter(unsigned position) const;
    float rotationOfCharacter(unsigned position) const;
    FloatRect extentOfCharacter(unsigned position) const;
    int characterNumberAtPosition(const FloatPoint&) const;

    // Public helper struct. Private classes in SVGTextQuery inherit from it.
    struct Data;

private:
    typedef bool (SVGTextQuery::*ProcessTextFragmentCallback)(Data*, const SVGTextFragment&) const;
    bool executeQuery(Data*, ProcessTextFragmentCallback) const;

    void collectTextBoxesInInlineBox(InlineIterator::InlineBoxIterator);
    bool mapStartEndPositionsIntoFragmentCoordinates(Data*, const SVGTextFragment&, unsigned& startPosition, unsigned& endPosition) const;
    void modifyStartEndPositionsRespectingLigatures(Data*, const SVGTextFragment&, unsigned& startPosition, unsigned& endPosition) const;

private:
    bool numberOfCharactersCallback(Data*, const SVGTextFragment&) const;
    bool textLengthCallback(Data*, const SVGTextFragment&) const;
    bool subStringLengthCallback(Data*, const SVGTextFragment&) const;
    bool startPositionOfCharacterCallback(Data*, const SVGTextFragment&) const;
    bool endPositionOfCharacterCallback(Data*, const SVGTextFragment&) const;
    bool rotationOfCharacterCallback(Data*, const SVGTextFragment&) const;
    bool extentOfCharacterCallback(Data*, const SVGTextFragment&) const;
    bool characterNumberAtPositionCallback(Data*, const SVGTextFragment&) const;

private:
    Vector<InlineIterator::SVGTextBoxIterator> m_textBoxes;
};

} // namespace WebCore
