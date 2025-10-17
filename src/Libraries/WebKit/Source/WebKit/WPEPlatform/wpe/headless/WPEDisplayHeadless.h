/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 7, 2025.
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
#ifndef WPEDisplayHeadless_h
#define WPEDisplayHeadless_h

#if !defined(__WPE_HEADLESS_H_INSIDE__) && !defined(BUILDING_WEBKIT)
#error "Only <wpe/headless/wpe-headless.h> can be included directly."
#endif

#include <glib-object.h>
#include <wpe/wpe-platform.h>

G_BEGIN_DECLS

#define WPE_TYPE_DISPLAY_HEADLESS (wpe_display_headless_get_type())
WPE_API G_DECLARE_FINAL_TYPE (WPEDisplayHeadless, wpe_display_headless, WPE, DISPLAY_HEADLESS, WPEDisplay)

WPE_API WPEDisplay *wpe_display_headless_new            (void);
WPE_API WPEDisplay *wpe_display_headless_new_for_device (const char *name,
                                                         GError    **error);

G_END_DECLS

#endif /* WPEDisplayHeadless_h */
