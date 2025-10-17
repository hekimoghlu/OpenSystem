/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 2, 2023.
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
#include "WPEInputMethodContextNone.h"

#include <wtf/glib/WTFGType.h>

struct _WPEInputMethodContextNonePrivate {
};

WEBKIT_DEFINE_FINAL_TYPE(WPEInputMethodContextNone, wpe_input_method_context_none, WPE_TYPE_INPUT_METHOD_CONTEXT, WPEInputMethodContext)

static void wpeInputMethodContextNoneGetPreeditString(WPEInputMethodContext*, char** text, GList** underlines, guint* cursorOffset)
{
    if (text)
        *text = g_strdup("");
    if (underlines)
        *underlines = nullptr;
    if (cursorOffset)
        *cursorOffset = 0;
}

static void wpe_input_method_context_none_class_init(WPEInputMethodContextNoneClass* klass)
{
    WPEInputMethodContextClass* imContextClass = WPE_INPUT_METHOD_CONTEXT_CLASS(klass);
    imContextClass->get_preedit_string = wpeInputMethodContextNoneGetPreeditString;
}

WPEInputMethodContext* wpeInputMethodContextNoneNew()
{
    return WPE_INPUT_METHOD_CONTEXT(g_object_new(WPE_TYPE_INPUT_METHOD_CONTEXT_NONE, nullptr));
}
