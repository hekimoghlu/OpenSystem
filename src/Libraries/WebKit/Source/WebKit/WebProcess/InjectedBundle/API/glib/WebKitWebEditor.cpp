/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 17, 2022.
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
#include "WebKitWebEditor.h"

#include "APIInjectedBundleEditorClient.h"
#include "WebKitWebEditorPrivate.h"
#include "WebKitWebPagePrivate.h"
#include <wtf/glib/WTFGType.h>

using namespace WebKit;
using namespace WebCore;

/**
 * WebKitWebEditor:
 * @See_also: #WebKitWebPage
 *
 * Access to editing capabilities of a #WebKitWebPage.
 *
 * The WebKitWebEditor provides access to various editing capabilities of
 * a #WebKitWebPage such as a possibility to react to the current selection in
 * #WebKitWebPage.
 *
 * Since: 2.10
 */
enum {
    SELECTION_CHANGED,

    LAST_SIGNAL
};

struct _WebKitWebEditorPrivate {
    WebKitWebPage* webPage;
};

static std::array<unsigned, LAST_SIGNAL> signals;

WEBKIT_DEFINE_FINAL_TYPE(WebKitWebEditor, webkit_web_editor, G_TYPE_OBJECT, GObject)

static void webkit_web_editor_class_init(WebKitWebEditorClass* klass)
{
    /**
     * WebKitWebEditor::selection-changed:
     * @editor: the #WebKitWebEditor on which the signal is emitted
     *
     * This signal is emitted for every selection change inside a #WebKitWebPage
     * as well as for every caret position change as the caret is a collapsed
     * selection.
     *
     * Since: 2.10
     */
    signals[SELECTION_CHANGED] = g_signal_new(
        "selection-changed",
        G_TYPE_FROM_CLASS(klass),
        G_SIGNAL_RUN_LAST,
        0, nullptr, nullptr,
        g_cclosure_marshal_VOID__VOID,
        G_TYPE_NONE, 0);
}

class PageEditorClient final : public API::InjectedBundle::EditorClient {
public:
    explicit PageEditorClient(WebKitWebEditor* editor)
        : m_editor(editor)
    {
    }

private:
    void didChangeSelection(WebPage&, const String&) final
    {
        g_signal_emit(m_editor, signals[SELECTION_CHANGED], 0);
    }

    WebKitWebEditor* m_editor;
};

WebKitWebEditor* webkitWebEditorCreate(WebKitWebPage* webPage)
{
    WebKitWebEditor* editor = WEBKIT_WEB_EDITOR(g_object_new(WEBKIT_TYPE_WEB_EDITOR, nullptr));
    editor->priv->webPage = webPage;
    webkitWebPageGetPage(webPage)->setInjectedBundleEditorClient(makeUnique<PageEditorClient>(editor));
    return editor;
}

/**
 * webkit_web_editor_get_page:
 * @editor: a #WebKitWebEditor
 *
 * Gets the #WebKitWebPage that is associated with the #WebKitWebEditor.
 *
 * Returns: (transfer none): the associated #WebKitWebPage
 *
 * Since: 2.10
 */
WebKitWebPage* webkit_web_editor_get_page(WebKitWebEditor* editor)
{
    g_return_val_if_fail(WEBKIT_IS_WEB_EDITOR(editor), nullptr);

    return editor->priv->webPage;
}
