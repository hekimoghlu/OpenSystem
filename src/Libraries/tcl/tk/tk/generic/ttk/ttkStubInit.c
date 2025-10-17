/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 31, 2025.
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
#include "tk.h"
#include "ttkTheme.h"

MODULE_SCOPE const TtkStubs ttkStubs;

/* !BEGIN!: Do not edit below this line. */

const TtkStubs ttkStubs = {
    TCL_STUB_MAGIC,
    TTK_STUBS_EPOCH,
    TTK_STUBS_REVISION,
    0,
    Ttk_GetTheme, /* 0 */
    Ttk_GetDefaultTheme, /* 1 */
    Ttk_GetCurrentTheme, /* 2 */
    Ttk_CreateTheme, /* 3 */
    Ttk_RegisterCleanup, /* 4 */
    Ttk_RegisterElementSpec, /* 5 */
    Ttk_RegisterElement, /* 6 */
    Ttk_RegisterElementFactory, /* 7 */
    Ttk_RegisterLayout, /* 8 */
    0, /* 9 */
    Ttk_GetStateSpecFromObj, /* 10 */
    Ttk_NewStateSpecObj, /* 11 */
    Ttk_GetStateMapFromObj, /* 12 */
    Ttk_StateMapLookup, /* 13 */
    Ttk_StateTableLookup, /* 14 */
    0, /* 15 */
    0, /* 16 */
    0, /* 17 */
    0, /* 18 */
    0, /* 19 */
    Ttk_GetPaddingFromObj, /* 20 */
    Ttk_GetBorderFromObj, /* 21 */
    Ttk_GetStickyFromObj, /* 22 */
    Ttk_MakePadding, /* 23 */
    Ttk_UniformPadding, /* 24 */
    Ttk_AddPadding, /* 25 */
    Ttk_RelievePadding, /* 26 */
    Ttk_MakeBox, /* 27 */
    Ttk_BoxContains, /* 28 */
    Ttk_PackBox, /* 29 */
    Ttk_StickBox, /* 30 */
    Ttk_AnchorBox, /* 31 */
    Ttk_PadBox, /* 32 */
    Ttk_ExpandBox, /* 33 */
    Ttk_PlaceBox, /* 34 */
    Ttk_NewBoxObj, /* 35 */
    0, /* 36 */
    0, /* 37 */
    0, /* 38 */
    0, /* 39 */
    Ttk_GetOrientFromObj, /* 40 */
};

/* !END!: Do not edit above this line. */
