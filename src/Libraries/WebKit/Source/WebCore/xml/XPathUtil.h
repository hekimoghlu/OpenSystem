/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 13, 2025.
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

#include <unicode/utypes.h>
#include <wtf/Forward.h>

namespace WebCore {

    class Node;

    namespace XPath {

        /* @return whether the given node is the root node */
        bool isRootDomNode(Node*);

        /* @return the 'string-value' of the given node as specified by http://www.w3.org/TR/xpath */
        String stringValue(Node*);

        /* @return whether the given node is a valid context node */
        bool isValidContextNode(Node&);

    } // namespace XPath

} // namespace WebCore
