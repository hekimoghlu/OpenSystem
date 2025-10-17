/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 31, 2024.
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
#ifndef WPEScreenDRM_h
#define WPEScreenDRM_h

#if !defined(__WPE_DRM_H_INSIDE__) && !defined(BUILDING_WEBKIT)
#error "Only <wpe/drm/wpe-drm.h> can be included directly."
#endif

#include <glib-object.h>
#include <wpe/wpe-platform.h>

G_BEGIN_DECLS

#define WPE_TYPE_SCREEN_DRM (wpe_screen_drm_get_type())
WPE_API G_DECLARE_FINAL_TYPE (WPEScreenDRM, wpe_screen_drm, WPE, SCREEN_DRM, WPEScreen)

WPE_API guint wpe_screen_drm_get_crtc_index(WPEScreenDRM *screen);

G_END_DECLS

#endif /* WPEScreenDRM_h */
