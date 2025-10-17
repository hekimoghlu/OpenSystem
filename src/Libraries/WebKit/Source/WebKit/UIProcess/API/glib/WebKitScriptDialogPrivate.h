/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 25, 2023.
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

#include "WebKitScriptDialog.h"
#include <wtf/text/CString.h>
#include <wtf/text/WTFString.h>

struct _WebKitScriptDialog {
    _WebKitScriptDialog(unsigned type, const CString& message, const CString& defaultText, Function<void(bool, const String&)>&& completionHandler)
        : type(type)
        , message(message)
        , defaultText(defaultText)
        , completionHandler(WTFMove(completionHandler))
    {
    }

    unsigned type;
    CString message;
    CString defaultText;

    bool confirmed { false };
    CString text;

    Function<void(bool, const String&)> completionHandler;

#if PLATFORM(GTK)
    GtkWidget* nativeDialog { nullptr };
#endif

#if PLATFORM(WPE)
    bool isUserHandled { true };
#endif

    int referenceCount { 1 };
};

WebKitScriptDialog* webkitScriptDialogCreate(unsigned type, const CString& message, const CString& defaultText, Function<void(bool, const String&)>&& completionHandler);
bool webkitScriptDialogIsRunning(WebKitScriptDialog*);
void webkitScriptDialogAccept(WebKitScriptDialog*);
void webkitScriptDialogDismiss(WebKitScriptDialog*);
void webkitScriptDialogSetUserInput(WebKitScriptDialog*, const String&);
bool webkitScriptDialogIsUserHandled(WebKitScriptDialog *);
