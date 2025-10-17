/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 7, 2022.
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

#include <wtf/glib/GRefPtr.h>

typedef struct _GtkWidgetPath GtkWidgetPath;
typedef struct _SecretValue SecretValue;
typedef struct _GskRenderNode GskRenderNode;

namespace WTF {

#if !USE(GTK4)
template <> GtkTargetList* refGPtr(GtkTargetList* ptr);
template <> void derefGPtr(GtkTargetList* ptr);
#endif

#if USE(LIBSECRET)
template <> SecretValue* refGPtr(SecretValue* ptr);
template <> void derefGPtr(SecretValue* ptr);
#endif

#if !USE(GTK4)
template <> GtkWidgetPath* refGPtr(GtkWidgetPath* ptr);
template <> void derefGPtr(GtkWidgetPath* ptr);
#endif

#if USE(GTK4)
template <> GskRenderNode* refGPtr(GskRenderNode* ptr);
template <> void derefGPtr(GskRenderNode* ptr);

template <> GdkEvent* refGPtr(GdkEvent* ptr);
template <> void derefGPtr(GdkEvent* ptr);
#endif

}

