/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 28, 2021.
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
#include "Text.h"

#include "Event.h"
#include "RenderCombineText.h"
#include "RenderSVGInlineText.h"
#include "RenderText.h"
#include "SVGElementInlines.h"
#include "SVGNames.h"
#include "ScopedEventQueue.h"
#include "ShadowRoot.h"
#include "StyleInheritedData.h"
#include "StyleResolver.h"
#include "StyleUpdate.h"
#include "TextManipulationController.h"
#include "TextNodeTraversal.h"
#include <wtf/CheckedArithmetic.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/CString.h>
#include <wtf/text/StringBuilder.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(Text);

Ref<Text> Text::create(Document& document, String&& data)
{
    return adoptRef(*new Text(document, WTFMove(data), TEXT_NODE, { }));
}

Ref<Text> Text::createEditingText(Document& document, String&& data)
{
    auto node = adoptRef(*new Text(document, WTFMove(data), TEXT_NODE, { }));
    node->setStateFlag(StateFlag::IsSpecialInternalNode);
    ASSERT(node->isEditingText());
    return node;
}

Text::~Text() = default;

ExceptionOr<Ref<Text>> Text::splitText(unsigned offset)
{
    if (offset > length())
        return Exception { ExceptionCode::IndexSizeError };

    EventQueueScope scope;
    auto oldData = data();
    Ref newText = virtualCreate(oldData.substring(offset));
    setDataWithoutUpdate(oldData.left(offset));

    dispatchModifiedEvent(oldData);

    if (RefPtr parent = parentNode()) {
        auto insertResult = parent->insertBefore(newText, protectedNextSibling());
        if (insertResult.hasException())
            return insertResult.releaseException();
    }

    protectedDocument()->textNodeSplit(*this);

    updateRendererAfterContentChange(0, oldData.length());

    return newText;
}

static const Text* earliestLogicallyAdjacentTextNode(const Text* text)
{
    const Node* node = text;
    while ((node = node->previousSibling())) {
        if (auto* maybeText = dynamicDowncast<Text>(*node))
            text = maybeText;
        else
            break;
    }
    return text;
}

static const Text* latestLogicallyAdjacentTextNode(const Text* text)
{
    const Node* node = text;
    while ((node = node->nextSibling())) {
        if (auto* maybeText = dynamicDowncast<Text>(*node))
            text = maybeText;
        else
            break;
    }
    return text;
}

String Text::wholeText() const
{
    const Text* startText = earliestLogicallyAdjacentTextNode(this);
    const Text* endText = latestLogicallyAdjacentTextNode(this);
    ASSERT(endText);
    const Node* onePastEndText = TextNodeTraversal::nextSibling(*endText);

    StringBuilder result;
    for (const Text* text = startText; text != onePastEndText; text = TextNodeTraversal::nextSibling(*text))
        result.append(text->data());
    return result.toString();
}

void Text::replaceWholeText(const String& newText)
{
    // Protect startText and endText against mutation event handlers removing the last ref
    RefPtr startText = const_cast<Text*>(earliestLogicallyAdjacentTextNode(this));
    RefPtr endText = const_cast<Text*>(latestLogicallyAdjacentTextNode(this));

    RefPtr parent = parentNode(); // Protect against mutation handlers moving this node during traversal
    for (RefPtr<Node> node = WTFMove(startText); is<Text>(node) && node != this && node->parentNode() == parent;) {
        Ref nodeToRemove = node.releaseNonNull();
        node = nodeToRemove->nextSibling();
        parent->removeChild(nodeToRemove);
    }

    if (this != endText) {
        RefPtr nodePastEndText = endText->nextSibling();
        for (RefPtr node = nextSibling(); is<Text>(node) && node != nodePastEndText && node->parentNode() == parent;) {
            Ref nodeToRemove = node.releaseNonNull();
            node = nodeToRemove->nextSibling();
            parent->removeChild(nodeToRemove);
        }
    }

    if (newText.isEmpty()) {
        if (parent && parentNode() == parent)
            parent->removeChild(*this);
        return;
    }

    setData(newText);
}

String Text::nodeName() const
{
    return "#text"_s;
}

Ref<Node> Text::cloneNodeInternal(TreeScope& treeScope, CloningOperation)
{
    Ref document = treeScope.documentScope();
    return create(document, String { data() });
}

static bool isSVGShadowText(const Text& text)
{
    ASSERT(text.parentNode());
    auto* parentShadowRoot = dynamicDowncast<ShadowRoot>(*text.parentNode());
    return parentShadowRoot && parentShadowRoot->host()->hasTagName(SVGNames::trefTag);
}

static bool isSVGText(const Text& text)
{
    ASSERT(text.parentNode());
    auto* parentElement = dynamicDowncast<SVGElement>(*text.parentNode());
    return parentElement && !parentElement->hasTagName(SVGNames::foreignObjectTag);
}

RenderPtr<RenderText> Text::createTextRenderer(const RenderStyle& style)
{
    if (isSVGText(*this) || isSVGShadowText(*this))
        return createRenderer<RenderSVGInlineText>(*this, data());

    if (style.hasTextCombine())
        return createRenderer<RenderCombineText>(*this, data());

    return createRenderer<RenderText>(RenderObject::Type::Text, *this, data());
}

Ref<Text> Text::virtualCreate(String&& data)
{
    return create(protectedDocument(), WTFMove(data));
}

void Text::updateRendererAfterContentChange(unsigned offsetOfReplacedData, unsigned lengthOfReplacedData)
{
    if (!isConnected())
        return;

    if (hasInvalidRenderer())
        return;

    protectedDocument()->updateTextRenderer(*this, offsetOfReplacedData, lengthOfReplacedData);
}

static void appendTextRepresentation(StringBuilder& builder, const Text& text)
{
    String value = text.data();
    builder.append(" length="_s, value.length());

    value = makeStringByReplacingAll(value, '\\', "\\\\"_s);
    value = makeStringByReplacingAll(value, '\n', "\\n"_s);
    
    constexpr size_t maxDumpLength = 30;
    if (value.length() > maxDumpLength)
        builder.append(" \""_s, StringView(value).left(maxDumpLength - 10), "...\""_s);
    else
        builder.append(" \""_s, value, '\"');
}

String Text::description() const
{
    StringBuilder builder;

    builder.append(CharacterData::description());
    appendTextRepresentation(builder, *this);

    return builder.toString();
}

String Text::debugDescription() const
{
    StringBuilder builder;

    builder.append(CharacterData::debugDescription());
    appendTextRepresentation(builder, *this);

    return builder.toString();
}

void Text::setDataAndUpdate(const String& newData, unsigned offsetOfReplacedData, unsigned oldLength, unsigned newLength, UpdateLiveRanges updateLiveRanges)
{
    auto oldData = data();
    CharacterData::setDataAndUpdate(newData, offsetOfReplacedData, oldLength, newLength, updateLiveRanges);

    // FIXME: Does not seem correct to do this for 0 offset only.
    if (!offsetOfReplacedData) {
        Ref document = this->document();
        CheckedPtr textManipulationController = document->textManipulationControllerIfExists();
        if (UNLIKELY(textManipulationController && oldData != newData))
            textManipulationController->didUpdateContentForNode(*this);
    }
}

} // namespace WebCore
