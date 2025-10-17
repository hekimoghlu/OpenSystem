/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 2, 2023.
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

namespace WebCore {

class Comment final : public CharacterData {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(Comment);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(Comment);
public:
    static Ref<Comment> create(Document&, String&&);

private:
    Comment(Document&, String&&);

    String nodeName() const override;
    Ref<Node> cloneNodeInternal(TreeScope&, CloningOperation) override;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::Comment)
    static bool isType(const WebCore::Node& node) { return node.nodeType() == WebCore::Node::COMMENT_NODE; }
SPECIALIZE_TYPE_TRAITS_END()
