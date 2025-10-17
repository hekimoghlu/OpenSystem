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
#pragma once

#include "DOMCSSCustomPropertyDescriptor.h"
#include "ExceptionOr.h"
#include "Supplementable.h"

namespace WebCore {

class Document;
class DOMCSSNamespace;

DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER(DOMCSSRegisterCustomProperty);
class DOMCSSRegisterCustomProperty final : public Supplement<DOMCSSNamespace> {
    WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(DOMCSSRegisterCustomProperty);
public:
    explicit DOMCSSRegisterCustomProperty(DOMCSSNamespace&) { }

    static ExceptionOr<void> registerProperty(Document&, const DOMCSSCustomPropertyDescriptor&);

private:
    static DOMCSSRegisterCustomProperty* from(DOMCSSNamespace&);
    static ASCIILiteral supplementName();
};

}
