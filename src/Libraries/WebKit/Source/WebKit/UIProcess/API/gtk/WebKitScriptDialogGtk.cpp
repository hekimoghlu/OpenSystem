/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 2, 2025.
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

#include "WebKitScriptDialogImpl.h"
#include "WebKitScriptDialogPrivate.h"

void webkitScriptDialogAccept(WebKitScriptDialog* scriptDialog)
{
    if (!WEBKIT_IS_SCRIPT_DIALOG_IMPL(scriptDialog->nativeDialog))
        return;
    webkitScriptDialogImplConfirm(WEBKIT_SCRIPT_DIALOG_IMPL(scriptDialog->nativeDialog));
}

void webkitScriptDialogDismiss(WebKitScriptDialog* scriptDialog)
{
    if (!WEBKIT_IS_SCRIPT_DIALOG_IMPL(scriptDialog->nativeDialog))
        return;
    webkitScriptDialogImplCancel(WEBKIT_SCRIPT_DIALOG_IMPL(scriptDialog->nativeDialog));
}

void webkitScriptDialogSetUserInput(WebKitScriptDialog* scriptDialog, const String& userInput)
{
    if (!WEBKIT_IS_SCRIPT_DIALOG_IMPL(scriptDialog->nativeDialog))
        return;

    webkitScriptDialogImplSetEntryText(WEBKIT_SCRIPT_DIALOG_IMPL(scriptDialog->nativeDialog), userInput);
}

bool webkitScriptDialogIsUserHandled(WebKitScriptDialog* scriptDialog)
{
    return !WEBKIT_IS_SCRIPT_DIALOG_IMPL(scriptDialog->nativeDialog);
}
