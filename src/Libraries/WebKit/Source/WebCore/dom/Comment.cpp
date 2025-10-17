/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 11, 2024.
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
#include "Comment.h"

#include "Document.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(Comment);

inline Comment::Comment(Document& document, String&& text)
    : CharacterData(document, WTFMove(text), COMMENT_NODE)
{
}

Ref<Comment> Comment::create(Document& document, String&& text)
{
    return adoptRef(*new Comment(document, WTFMove(text)));
}

String Comment::nodeName() const
{
    return "#comment"_s;
}

Ref<Node> Comment::cloneNodeInternal(TreeScope& treeScope, CloningOperation)
{
    Ref document = treeScope.documentScope();
    return create(document, String { data() });
}

} // namespace WebCore
