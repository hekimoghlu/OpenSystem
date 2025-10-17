/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 7, 2022.
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
/*
 *
 * (C) 2002,2005 by Marcin Dalecki <martin@dalecki.de>
 *
 * MARCIN DALECKI ASSUMES NO RESPONSIBILITY FOR THE USE OR INABILITY TO USE ANY
 * OF THIS SOFTWARE . THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY
 * KIND, AND MARCIN DALECKI EXPRESSLY DISCLAIMS ALL IMPLIED WARRANTIES,
 * INCLUDING BUT NOT LIMITED TO THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef EnhancedB_H
#define EnhancedB_H

/*
 * New resources for the Extended Pushbutton widget
 */

#ifndef XmNshift
# define XmNshift		"shift"
#endif
#ifndef XmCShift
# define XmCShift		"Shift"
#endif

#ifndef XmNlabelLocation
# define XmNlabelLocation	"labelLocation"
#endif
#ifndef XmCLocation
# define XmCLocation		"Location"
#endif

#ifndef XmNpixmapData
# define XmNpixmapData		"pixmapData"
#endif

#ifndef XmNpixmapFile
# define XmNpixmapFile		"pixmapFile"
#endif

/*
 * Constants for labelLocation.
 */
#ifdef HAVE_XM_JOINSIDET_H
# include <Xm/JoinSideT.h>
#else
# define XmLEFT	    1
# define XmRIGHT    2
# define XmTOP	    3
# define XmBOTTOM   4
#endif

#define XmIsEnhancedButton(w) XtIsSubclass(w, xmEnhancedButtonWidgetClass)

/*
 * Convenience creation function.
 */
extern Widget XgCreateEPushButtonWidget(Widget, char *, ArgList, Cardinal);

extern WidgetClass xmEnhancedButtonWidgetClass;
typedef struct _XmEnhancedButtonClassRec *XmEnhancedButtonWidgetClass;
typedef struct _XmEnhancedButtonRec *XmEnhancedButtonWidget;

#endif
