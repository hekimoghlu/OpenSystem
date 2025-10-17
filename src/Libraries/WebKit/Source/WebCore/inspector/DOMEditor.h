/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 29, 2022.
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

#include <wtf/TZoneMalloc.h>

namespace WebCore {

class ContainerNode;
class Element;
class InspectorHistory;
class Node;
class Text;

typedef String ErrorString;

class DOMEditor {
    WTF_MAKE_TZONE_ALLOCATED(DOMEditor);
    WTF_MAKE_NONCOPYABLE(DOMEditor);
public:
    explicit DOMEditor(InspectorHistory&);
    ~DOMEditor();

    ExceptionOr<void> insertBefore(ContainerNode& parentNode, Ref<Node>&&, Node* anchorNode);
    ExceptionOr<void> removeChild(ContainerNode& parentNode, Node&);
    ExceptionOr<void> setAttribute(Element&, const AtomString& name, const AtomString& value);
    ExceptionOr<void> removeAttribute(Element&, const AtomString& name);
    ExceptionOr<void> setOuterHTML(Node&, const String& html, Node*& newNode);
    ExceptionOr<void> replaceWholeText(Text&, const String& text);
    ExceptionOr<void> replaceChild(ContainerNode& parentNode, Ref<Node>&& newNode, Node& oldNode);
    ExceptionOr<void> setNodeValue(Node&, const String& value);
    ExceptionOr<void> insertAdjacentHTML(Element&, const String& where, const String& html);

    bool insertBefore(ContainerNode& parentNode, Ref<Node>&&, Node* anchorNode, ErrorString&);
    bool removeChild(ContainerNode& parentNode, Node&, ErrorString&);
    bool setAttribute(Element&, const AtomString& name, const AtomString& value, ErrorString&);
    bool removeAttribute(Element&, const AtomString& name, ErrorString&);
    bool setOuterHTML(Node&, const String& html, Node*& newNode, ErrorString&);
    bool replaceWholeText(Text&, const String& text, ErrorString&);
    bool insertAdjacentHTML(Element&, const String& where, const String& html, ErrorString&);

private:
    class DOMAction;
    class InsertAdjacentHTMLAction;
    class InsertBeforeAction;
    class RemoveAttributeAction;
    class RemoveChildAction;
    class ReplaceChildNodeAction;
    class ReplaceWholeTextAction;
    class SetAttributeAction;
    class SetNodeValueAction;
    class SetOuterHTMLAction;

    InspectorHistory& m_history;
};

} // namespace WebCore
