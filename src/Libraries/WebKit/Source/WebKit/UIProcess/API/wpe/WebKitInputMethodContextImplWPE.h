/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 5, 2023.
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

#include "WebKitInputMethodContext.h"

#if ENABLE(WPE_PLATFORM)

#include <glib-object.h>

typedef struct _WPEView WPEView;

G_BEGIN_DECLS

#define WEBKIT_TYPE_INPUT_METHOD_CONTEXT_IMPL_WPE (webkit_input_method_context_impl_wpe_get_type())
G_DECLARE_FINAL_TYPE (WebKitInputMethodContextImplWPE, webkit_input_method_context_impl_wpe, WEBKIT, INPUT_METHOD_CONTEXT_IMPL_WPE, WebKitInputMethodContext)

WebKitInputMethodContext* webkitInputMethodContextImplWPENew(WPEView *view);

G_END_DECLS

#endif // ENABLE(WPE_PLATFORM)
