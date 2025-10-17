/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 26, 2022.
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
#include "WebKitScriptDialog.h"

#include "WebKitScriptDialogPrivate.h"

// Callbacks invoked by WebDriver commands
// As WPE has currently no public API to allow the browser to respond to these commands,
// we mimic the expected behavior in these callbacks like one would do in a reference browser.
void webkitScriptDialogAccept(WebKitScriptDialog* dialog)
{
    auto dialog_type = webkit_script_dialog_get_dialog_type(dialog);
    if (dialog_type == WEBKIT_SCRIPT_DIALOG_CONFIRM || dialog_type == WEBKIT_SCRIPT_DIALOG_BEFORE_UNLOAD_CONFIRM)
        webkit_script_dialog_confirm_set_confirmed(dialog, TRUE);
    // W3C WebDriver tests expect an empty string instead of a null one when the prompt is accepted.
    if (dialog_type == WEBKIT_SCRIPT_DIALOG_PROMPT && dialog->text.isNull())
        webkit_script_dialog_prompt_set_text(dialog, dialog->defaultText.isNull() ? "" : dialog->defaultText.data());
    webkit_script_dialog_unref(dialog);
}

void webkitScriptDialogDismiss(WebKitScriptDialog* dialog)
{
    auto dialog_type = webkit_script_dialog_get_dialog_type(dialog);
    if (dialog_type == WEBKIT_SCRIPT_DIALOG_CONFIRM || dialog_type == WEBKIT_SCRIPT_DIALOG_BEFORE_UNLOAD_CONFIRM)
        webkit_script_dialog_confirm_set_confirmed(dialog, FALSE);
    webkit_script_dialog_unref(dialog);
}

void webkitScriptDialogSetUserInput(WebKitScriptDialog* dialog, const String& input)
{
    if (webkit_script_dialog_get_dialog_type(dialog) != WEBKIT_SCRIPT_DIALOG_PROMPT)
        return;

    webkit_script_dialog_prompt_set_text(dialog, input.utf8().data());
}

bool webkitScriptDialogIsUserHandled(WebKitScriptDialog* dialog)
{
    return dialog->isUserHandled;
}
