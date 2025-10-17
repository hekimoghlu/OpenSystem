/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 6, 2022.
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

#include "HTMLTableCellElement.h"
#include "HTMLTablePartElement.h"

namespace WebCore {

class HTMLTableRowElement final : public HTMLTablePartElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(HTMLTableRowElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(HTMLTableRowElement);
public:
    static Ref<HTMLTableRowElement> create(Document&);
    static Ref<HTMLTableRowElement> create(const QualifiedName&, Document&);

    WEBCORE_EXPORT int rowIndex() const;
    void setRowIndex(int);

    WEBCORE_EXPORT int sectionRowIndex() const;
    void setSectionRowIndex(int);

    WEBCORE_EXPORT ExceptionOr<Ref<HTMLTableCellElement>> insertCell(int index = -1);
    WEBCORE_EXPORT ExceptionOr<void> deleteCell(int index);

    WEBCORE_EXPORT Ref<HTMLCollection> cells();

private:
    HTMLTableRowElement(const QualifiedName&, Document&);
};

} // namespace
