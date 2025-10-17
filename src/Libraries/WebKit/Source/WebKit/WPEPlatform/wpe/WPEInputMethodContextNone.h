/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 17, 2024.
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

#include <glib-object.h>
#include <wpe/wpe-platform.h>

G_BEGIN_DECLS

#define WPE_TYPE_INPUT_METHOD_CONTEXT_NONE (wpe_input_method_context_none_get_type())
G_DECLARE_FINAL_TYPE(WPEInputMethodContextNone, wpe_input_method_context_none, WPE, INPUT_METHOD_CONTEXT_NONE, WPEInputMethodContext)

WPEInputMethodContext* wpeInputMethodContextNoneNew();

G_END_DECLS
