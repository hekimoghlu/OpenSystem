/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 24, 2025.
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
#ifndef WPEDisplayQtQuick_h
#define WPEDisplayQtQuick_h

#include <glib-object.h>
#include <wpe/wpe-platform.h>

G_BEGIN_DECLS

#define WPE_TYPE_DISPLAY_QTQUICK (wpe_display_qtquick_get_type())
G_DECLARE_FINAL_TYPE (WPEDisplayQtQuick, wpe_display_qtquick, WPE, DISPLAY_QTQUICK, WPEDisplay)

WPEDisplay *wpe_display_qtquick_new (void);

G_END_DECLS

#endif /* WPEDisplayQtQuick_h */
