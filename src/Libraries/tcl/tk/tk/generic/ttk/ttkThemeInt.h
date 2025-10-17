/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 24, 2022.
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
#ifndef _TTKTHEMEINT
#define _TTKTHEMEINT

#include "ttkTheme.h"

typedef struct Ttk_TemplateNode_ Ttk_TemplateNode, *Ttk_LayoutTemplate;

MODULE_SCOPE Ttk_ElementClass *Ttk_GetElement(Ttk_Theme, const char *name);
MODULE_SCOPE const char *Ttk_ElementClassName(Ttk_ElementClass *);

MODULE_SCOPE void Ttk_ElementSize(
	Ttk_ElementClass *, Ttk_Style, char *recordPtr, Tk_OptionTable,
	Tk_Window tkwin, Ttk_State state,
	int *widthPtr, int *heightPtr, Ttk_Padding*);
MODULE_SCOPE void Ttk_DrawElement(
	Ttk_ElementClass *, Ttk_Style, char *recordPtr, Tk_OptionTable,
	Tk_Window tkwin, Drawable d, Ttk_Box b, Ttk_State state);

MODULE_SCOPE Tcl_Obj *Ttk_QueryStyle(
    Ttk_Style, void *, Tk_OptionTable, const char *, Ttk_State state);

MODULE_SCOPE Ttk_LayoutTemplate Ttk_ParseLayoutTemplate(
	Tcl_Interp *, Tcl_Obj *);
MODULE_SCOPE Tcl_Obj *Ttk_UnparseLayoutTemplate(Ttk_LayoutTemplate);
MODULE_SCOPE Ttk_LayoutTemplate Ttk_BuildLayoutTemplate(Ttk_LayoutSpec);
MODULE_SCOPE void Ttk_FreeLayoutTemplate(Ttk_LayoutTemplate);
MODULE_SCOPE void Ttk_RegisterLayoutTemplate(
    Ttk_Theme theme, const char *layoutName, Ttk_LayoutTemplate);

MODULE_SCOPE Ttk_Style Ttk_GetStyle(Ttk_Theme themePtr, const char *styleName);
MODULE_SCOPE Ttk_LayoutTemplate Ttk_FindLayoutTemplate(
    Ttk_Theme themePtr, const char *layoutName);

MODULE_SCOPE const char *Ttk_StyleName(Ttk_Style);

#endif /* _TTKTHEMEINT */
