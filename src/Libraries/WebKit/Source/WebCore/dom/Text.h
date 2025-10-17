/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 9, 2024.
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

#include "CharacterData.h"
#include "RenderPtr.h"

namespace WebCore {

class RenderText;

class Text : public CharacterData {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(Text);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(Text);
public:
    static const unsigned defaultLengthLimit = 1 << 16;

    static Ref<Text> create(Document&, String&&);
    static Ref<Text> createEditingText(Document&, String&&);

    virtual ~Text();

    WEBCORE_EXPORT ExceptionOr<Ref<Text>> splitText(unsigned offset);

    // DOM Level 3: http://www.w3.org/TR/DOM-Level-3-Core/core.html#ID-1312295772

    WEBCORE_EXPORT String wholeText() const;
    WEBCORE_EXPORT void replaceWholeText(const String&);
    
    RenderPtr<RenderText> createTextRenderer(const RenderStyle&);
    
    bool canContainRangeEndPoint() const final { return true; }

    RenderText* renderer() const;

    void updateRendererAfterContentChange(unsigned offsetOfReplacedData, unsigned lengthOfReplacedData);

    String description() const final;
    String debugDescription() const final;

protected:
    Text(Document& document, String&& data, NodeType type, OptionSet<TypeFlag> typeFlags)
        : CharacterData(document, WTFMove(data), type, typeFlags | TypeFlag::IsText)
    {
        ASSERT(!isContainerNode());
    }

private:
    String nodeName() const override;
    Ref<Node> cloneNodeInternal(TreeScope&, CloningOperation) override;
    void setDataAndUpdate(const String&, unsigned offsetOfReplacedData, unsigned oldLength, unsigned newLength, UpdateLiveRanges) final;

    virtual Ref<Text> virtualCreate(String&&);
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::Text)
    static bool isType(const WebCore::Node& node) { return node.isTextNode(); }
SPECIALIZE_TYPE_TRAITS_END()
