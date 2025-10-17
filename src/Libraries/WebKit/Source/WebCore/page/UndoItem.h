/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 14, 2023.
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

#include "EventTarget.h"
#include "VoidCallback.h"
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class Document;
class UndoManager;

class UndoItem : public RefCountedAndCanMakeWeakPtr<UndoItem> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(UndoItem);
public:
    struct Init {
        String label;
        RefPtr<VoidCallback> undo;
        RefPtr<VoidCallback> redo;
    };

    static Ref<UndoItem> create(Init&& init)
    {
        return adoptRef(*new UndoItem(WTFMove(init)));
    }

    bool isValid() const;
    void invalidate();

    Document* document() const;
    RefPtr<Document> protectedDocument() const;

    UndoManager* undoManager() const;
    void setUndoManager(UndoManager*);

    const String& label() const { return m_label; }
    VoidCallback& undoHandler() const { return m_undoHandler.get(); }
    VoidCallback& redoHandler() const { return m_redoHandler.get(); }

private:
    UndoItem(Init&& init)
        : m_label(WTFMove(init.label))
        , m_undoHandler(init.undo.releaseNonNull())
        , m_redoHandler(init.redo.releaseNonNull())
    {
    }

    String m_label;
    Ref<VoidCallback> m_undoHandler;
    Ref<VoidCallback> m_redoHandler;
    WeakPtr<UndoManager> m_undoManager;
    WeakPtr<Document, WeakPtrImplWithEventTargetData> m_document;
};

} // namespace WebCore
