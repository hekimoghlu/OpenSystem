/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 15, 2022.
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
#include <tk.h>

#include "ttkTheme.h"
#include "ttkWidget.h"

/* +++ Separator widget record:
 */
typedef struct
{
    Tcl_Obj	*orientObj;
    int 	orient;
} SeparatorPart;

typedef struct
{
    WidgetCore core;
    SeparatorPart separator;
} Separator;

static Tk_OptionSpec SeparatorOptionSpecs[] =
{
    {TK_OPTION_STRING_TABLE, "-orient", "orient", "Orient", "horizontal",
	Tk_Offset(Separator,separator.orientObj),
	Tk_Offset(Separator,separator.orient),
	0,(ClientData)ttkOrientStrings,STYLE_CHANGED },

    WIDGET_INHERIT_OPTIONS(ttkCoreOptionSpecs)
};

/*
 * GetLayout hook --
 * 	Choose layout based on -orient option.
 */
static Ttk_Layout SeparatorGetLayout(
    Tcl_Interp *interp, Ttk_Theme theme, void *recordPtr)
{
    Separator *sep = recordPtr;
    return TtkWidgetGetOrientedLayout(
	interp, theme, recordPtr, sep->separator.orientObj);
}

/*
 * Widget commands:
 */
static const Ttk_Ensemble SeparatorCommands[] = {
    { "configure",	TtkWidgetConfigureCommand,0 },
    { "cget",		TtkWidgetCgetCommand,0 },
    { "identify",	TtkWidgetIdentifyCommand,0 },
    { "instate",	TtkWidgetInstateCommand,0 },
    { "state",  	TtkWidgetStateCommand,0 },
    { 0,0,0 }
};

/*
 * Widget specification:
 */
static WidgetSpec SeparatorWidgetSpec =
{
    "TSeparator",		/* className */
    sizeof(Separator),		/* recordSize */
    SeparatorOptionSpecs,	/* optionSpecs */
    SeparatorCommands,		/* subcommands */
    TtkNullInitialize,		/* initializeProc */
    TtkNullCleanup,		/* cleanupProc */
    TtkCoreConfigure,		/* configureProc */
    TtkNullPostConfigure,	/* postConfigureProc */
    SeparatorGetLayout,		/* getLayoutProc */
    TtkWidgetSize, 		/* sizeProc */
    TtkWidgetDoLayout,		/* layoutProc */
    TtkWidgetDisplay		/* displayProc */
};

TTK_BEGIN_LAYOUT(SeparatorLayout)
    TTK_NODE("Separator.separator", TTK_FILL_BOTH)
TTK_END_LAYOUT

/* +++ Sizegrip widget:
 * 	Has no options or methods other than the standard ones.
 */

static const Ttk_Ensemble SizegripCommands[] = {
    { "configure",	TtkWidgetConfigureCommand,0 },
    { "cget",		TtkWidgetCgetCommand,0 },
    { "identify",	TtkWidgetIdentifyCommand,0 },
    { "instate",	TtkWidgetInstateCommand,0 },
    { "state",  	TtkWidgetStateCommand,0 },
    { 0,0,0 }
};

static WidgetSpec SizegripWidgetSpec =
{
    "TSizegrip",		/* className */
    sizeof(WidgetCore),		/* recordSize */
    ttkCoreOptionSpecs, 	/* optionSpecs */
    SizegripCommands,		/* subcommands */
    TtkNullInitialize,		/* initializeProc */
    TtkNullCleanup,		/* cleanupProc */
    TtkCoreConfigure,		/* configureProc */
    TtkNullPostConfigure,	/* postConfigureProc */
    TtkWidgetGetLayout, 	/* getLayoutProc */
    TtkWidgetSize, 		/* sizeProc */
    TtkWidgetDoLayout,		/* layoutProc */
    TtkWidgetDisplay		/* displayProc */
};

TTK_BEGIN_LAYOUT(SizegripLayout)
    TTK_NODE("Sizegrip.sizegrip", TTK_PACK_BOTTOM|TTK_STICK_S|TTK_STICK_E)
TTK_END_LAYOUT

/* +++ Initialization:
 */

MODULE_SCOPE
void TtkSeparator_Init(Tcl_Interp *interp)
{
    Ttk_Theme theme = Ttk_GetDefaultTheme(interp);

    Ttk_RegisterLayout(theme, "TSeparator", SeparatorLayout);
    Ttk_RegisterLayout(theme, "TSizegrip", SizegripLayout);

    RegisterWidget(interp, "ttk::separator", &SeparatorWidgetSpec);
    RegisterWidget(interp, "ttk::sizegrip", &SizegripWidgetSpec);
}

/*EOF*/
