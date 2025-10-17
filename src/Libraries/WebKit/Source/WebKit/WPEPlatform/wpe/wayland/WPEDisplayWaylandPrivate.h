/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 11, 2024.
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

#include "WPEDisplayWayland.h"
#include "WPEScreen.h"
#include "WPEWaylandCursor.h"
#include "WPEWaylandSeat.h"

struct xdg_wm_base* wpeDisplayWaylandGetXDGWMBase(WPEDisplayWayland*);
WPE::WaylandSeat* wpeDisplayWaylandGetSeat(WPEDisplayWayland*);
WPE::WaylandCursor* wpeDisplayWaylandGetCursor(WPEDisplayWayland*);
WPEScreen* wpeDisplayWaylandFindScreen(WPEDisplayWayland*, struct wl_output*);
struct zwp_linux_dmabuf_v1* wpeDisplayWaylandGetLinuxDMABuf(WPEDisplayWayland*);
struct zwp_linux_explicit_synchronization_v1* wpeDisplayWaylandGetLinuxExplicitSync(WPEDisplayWayland*);
struct zwp_text_input_v1* wpeDisplayWaylandGetTextInputV1(WPEDisplayWayland*);
struct zwp_text_input_v3* wpeDisplayWaylandGetTextInputV3(WPEDisplayWayland*);
struct zwp_pointer_constraints_v1* wpeDisplayWaylandGetPointerConstraints(WPEDisplayWayland*);
struct zwp_relative_pointer_manager_v1* wpeDisplayWaylandGetRelativePointerManager(WPEDisplayWayland*);
