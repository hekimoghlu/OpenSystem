/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 21, 2022.
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

#include "CSSStyleSheet.h"
#include "ExceptionOr.h"
#include "JSObservableArray.h"

namespace WebCore {

class ContainerNode;

class CSSStyleSheetObservableArray : public JSC::ObservableArray {
public:
    static Ref<CSSStyleSheetObservableArray> create(ContainerNode& treeScope);

    ExceptionOr<void> setSheets(Vector<Ref<CSSStyleSheet>>&&);
    const Vector<Ref<CSSStyleSheet>>& sheets() const { return m_sheets; }

private:
    explicit CSSStyleSheetObservableArray(ContainerNode& treeScope);

    std::optional<Exception> shouldThrowWhenAddingSheet(const CSSStyleSheet&) const;

    // JSC::ObservableArray
    bool setValueAt(JSC::JSGlobalObject*, unsigned index, JSC::JSValue) final;
    void removeLast() final;
    JSC::JSValue valueAt(JSC::JSGlobalObject*, unsigned index) const final;
    unsigned length() const final { return m_sheets.size(); }
    void shrinkTo(unsigned) final;

    TreeScope* treeScope() const;

    void didAddSheet(CSSStyleSheet&);
    void willRemoveSheet(CSSStyleSheet&);

    WeakPtr<ContainerNode, WeakPtrImplWithEventTargetData> m_treeScope;
    Vector<Ref<CSSStyleSheet>> m_sheets;
};

} // namespace WebCore
