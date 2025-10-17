/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 21, 2023.
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
#include "DocumentType.h"

#include "Document.h"
#include "Element.h"
#include "NamedNodeMap.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(DocumentType);

DocumentType::DocumentType(Document& document, const String& name, const String& publicId, const String& systemId)
    : Node(document, DOCUMENT_TYPE_NODE, { })
    , m_name(name)
    , m_publicId(publicId.isNull() ? emptyString() : publicId)
    , m_systemId(systemId.isNull() ? emptyString() : systemId)
{
}

String DocumentType::nodeName() const
{
    return name();
}

Ref<Node> DocumentType::cloneNodeInternal(TreeScope& treeScope, CloningOperation)
{
    Ref document = treeScope.documentScope();
    return create(document, m_name, m_publicId, m_systemId);
}

}
